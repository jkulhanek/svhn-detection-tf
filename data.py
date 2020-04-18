import tensorflow as tf
from functools import partial
from svhn_dataset import SVHN
import utils
from augment import build_augment

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
    orig_classes = tf.cast(x['classes'], tf.int32)
    orig_bboxes = tf.cast(x['bboxes'], tf.float32)
    classes, bboxes, mask = utils.bboxes_training(anchors, orig_classes, orig_bboxes)
    onehot_classes = tf.one_hot(classes - 1, depth=SVHN.LABELS, dtype=tf.float32)
    onehot_classes = onehot_classes * tf.expand_dims(tf.cast(classes > 0, dtype=onehot_classes.dtype), -1)
    class_mask = mask
    regression_mask = tf.logical_and(class_mask, classes > 0)
    return { 'image':x['image'], 'bbox': bboxes, 'class': onehot_classes, 
            'class_mask': class_mask, 'regression_mask': regression_mask }

def generate_evaluation_data(x):
    orig_classes = tf.cast(x['classes'], tf.int32)
    orig_bboxes = tf.cast(x['bboxes'], tf.float32)
    return { 'image':x['image'], 'bbox': orig_bboxes, 'class': orig_classes }

def augment_map(bboxes, img, args):
    return augment(img, bboxes,
                                width_shift=args.aug_width_shift, height_shift=args.aug_height_shift,
                                zoom=args.aug_zoom,
                                rotation=args.aug_rotation,
                                vertical_fraction=args.aug_vertical_fraction,
                                horizontal_fraction=args.aug_horizontal_fraction)

def create_data(batch_size, anchors, image_size, test=False, augmentation='none'):
    assert test == False or batch_size <= 8 
    assert augmentation in ['none','retina','retina-rotation']
    dataset = SVHN()
    anchors = tf.cast(tf.convert_to_tensor(anchors), tf.float32)
    def create_dataset(x):
        if test:
            x = dataset.train.take(1)
        return x.map(SVHN.parse) \
                .map(scale_input(image_size)) \

    train, dev, test = tuple(map(create_dataset, 
        (dataset.train, dataset.dev, dataset.test)))

    augment = lambda x: x
    if augmentation == 'retina':
        augment = build_augment(False)
    elif augmentation == 'retina-rotate':
        augment = build_augment(True)


    # Generate training data with matched gt boxes
    train_dataset = train.map(augment).map(partial(generate_training_data, anchors)).shuffle(3000)
    dev_dataset = dev.map(partial(generate_training_data, anchors)).cache()

    # Generate evaluation data
    eval_dataset = dev.map(generate_evaluation_data).cache() 
    return (train_dataset, dev_dataset, eval_dataset)
