import tensorflow as tf
from functools import partial
from svhn_dataset import SVHN
import utils

def scale_input(image_size):
    def scale(x):
        x = dict(**x)
        shape = tf.shape(x['image'])
        oh, ow = image_size / shape[0], image_size / shape[1] 
        x['bboxes'] = tf.cast(x['bboxes'], tf.float32) * tf.convert_to_tensor([oh, ow, oh, ow], tf.float32)
        x['image'] = tf.image.resize(x['image'], (image_size, image_size))
        return x
    return scale

def generate_training_data(anchors, x):    
    classes, bboxes, mask = utils.bboxes_training(anchors, tf.cast(x['classes'], tf.int32), tf.cast(x['bboxes'], tf.float32))
    onehot_classes = tf.one_hot(classes, depth=SVHN.LABELS, dtype=tf.float32)
    class_mask = mask
    regression_mask = tf.logical_and(class_mask, classes > 0)
    return x['image'], { 'bbox': bboxes, 'class': onehot_classes, 
            'class_mask': class_mask, 'regression_mask': regression_mask}

def create_data(batch_size, anchors, image_size, test=False, evaluation=False):
    assert test == False or batch_size <= 8 
    dataset = SVHN()
    anchors = tf.cast(tf.convert_to_tensor(anchors), tf.float32)
    def create_dataset(x):
        if test:
            x = dataset.train.take(8)
        return x.map(SVHN.parse) \
                .map(scale_input(image_size)) \
                .map(partial(generate_training_data, anchors)) \
                .cache() 

    train, dev, test = tuple(map(create_dataset, 
        (dataset.train, dataset.dev, dataset.test)))

    if evaluation:
        train = train.shuffle(3000)
    return (train, dev, test)
