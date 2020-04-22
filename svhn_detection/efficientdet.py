import tensorflow as tf
from functools import partial
from efficientnet import pretrained_efficientnet_b0 
import numpy as np

class FastFusionWithActivation(tf.keras.layers.Layer):
    def __init__(self, ninputs, filters, **kwargs):
        super().__init__(**kwargs)
        self.bnidentity = tf.keras.layers.BatchNormalization()
        self.convidentity = tf.keras.layers.Conv2D(filters, 1, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.SeparableConv2D(filters, 3, padding='same')
        self.ff_weight = tf.Variable(
                name='ff_weight',
                initial_value=tf.keras.initializers.constant(1 / ninputs)(shape=(ninputs,)),
                trainable=True
        )
        self.ninputs = ninputs
        self.filters = filters

    def get_config(self): 
        config = super().get_config().copy()
        config.update({
            'ninputs': self.ninputs,
            'filters': self.filters,
        })
        return config

    def call(self, x, training=True):
        # Resize the last input to match the rest
        if x[-1].shape[1] > x[0].shape[1]: 
            x[-1] = tf.nn.max_pool2d(x[-1], 2, 2, 'SAME')
        elif x[-1].shape[1] < x[0].shape[1]:
            new_shape = tf.shape(x[0])[1:-1]
            x[-1] = tf.image.resize(x[-1], new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            #x[-1] = tf.keras.backend.resize_images(x[-1], 2, 2, 'channels_last', interpolation='nearest')
        x[-1] = self.bnidentity(self.convidentity(x[-1], training=training), training=training) 
        w = tf.nn.relu(self.ff_weight)
        w_sum = tf.math.reduce_sum(w) + 0.0001 # epsilon from the paper

        # Merge features
        x = sum([x[i] * w[i] for i in range(self.ninputs)])
        x /= w_sum
        x = self.conv(x, training=training)
        x = self.bn(x, training=training)
        x = tf.nn.swish(x)
        return x

def conv_change_filters(x, filters): 
    _, width, _, num_channels = x.get_shape().as_list()
    if num_channels != filters:
        x = tf.keras.layers.Conv2D(filters, 1)(x)
    return x

def build_BiFPNLayer(x, filters = 64, pyramid_levels=5, downsampling = True):
    ps = x[:pyramid_levels]

    # Upsampling part
    nps = x[:pyramid_levels]
    for i in range(1, pyramid_levels):
        nps[i] = FastFusionWithActivation(2, filters)([nps[i], nps[i - 1]])

    if not downsampling:
        return nps[-1]

    # Downsampling part
    for i in range(pyramid_levels - 2, 1, -1):
        nps[i] = FastFusionWithActivation(3, filters)([ps[i], nps[i], ps[i + 1]])
    nps[0] = FastFusionWithActivation(2, filters)([ps[0], nps[1]])
    return nps

def ConvFlatten(output_size):
    @tf.function(experimental_relax_shapes=True)
    def reshape(x):
        shape = tf.shape(x)
        n, h, w = shape[0], shape[1], shape[2] 
        x = tf.reshape(x, [n, h, w, -1, output_size])
        x = tf.reshape(x, [n, -1, output_size])
        return x 
    return tf.keras.layers.Lambda(reshape)

def DetectionHead(filters, num_classes, num_anchors):
    conv_channels = filters
    return tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(conv_channels, 3, use_bias=False, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(tf.nn.swish),
        tf.keras.layers.SeparableConv2D(conv_channels, 3, use_bias=False, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(tf.nn.swish),
        tf.keras.layers.SeparableConv2D(conv_channels, 3, use_bias=False, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(tf.nn.swish),
        tf.keras.layers.SeparableConv2D(
            num_classes * num_anchors, 3, padding='same',
            bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01))),
        ConvFlatten(num_classes)
    ])

def BoxHead(filters, num_anchors):
    conv_channels = filters
    return tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(conv_channels, 3, use_bias=False, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(tf.nn.swish),
        tf.keras.layers.SeparableConv2D(conv_channels, 3, use_bias=False, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(tf.nn.swish),
        tf.keras.layers.SeparableConv2D(conv_channels, 3, use_bias=False, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(tf.nn.swish),
        tf.keras.layers.SeparableConv2D(
            4 * num_anchors, 3, padding='same',
            bias_initializer=tf.zeros_initializer()),
        ConvFlatten(4)
    ])



def EfficientDet(num_classes, anchors_per_level, pyramid_levels = 4, backbone = None, input_size=224, filters = 64, num_layers = 3):
    fpn_channels = filters

    input_tensor = tf.keras.layers.Input((input_size, input_size, 3), dtype=tf.float32)
    if backbone is None:
        backbone = pretrained_efficientnet_b0(False, dynamic_shape=True, drop_connect_rate = 0.0)

    # Fix the number of output feature maps
    x = backbone(input_tensor)[1:-1][-pyramid_levels:]
    while len(x) < pyramid_levels:
        l = x[0]
        l = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(fpn_channels, 3, strides=2, use_bias=False, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.swish),
        ])(l)
        x.insert(0, l)

    x = list(map(partial(conv_change_filters, filters=fpn_channels), x))
    for _ in range(num_layers):
        x = build_BiFPNLayer(x, fpn_channels, pyramid_levels = pyramid_levels)
    class_predict = list(map(DetectionHead(fpn_channels, num_classes, anchors_per_level), x))
    class_predict = tf.keras.layers.Concatenate(axis=1, name='class')(class_predict)
    box_predict = list(map(BoxHead(fpn_channels, anchors_per_level), x))
    box_predict = tf.keras.layers.Concatenate(axis=1, name='box')(box_predict)
    return tf.keras.Model(inputs = input_tensor, outputs = [class_predict, box_predict])
