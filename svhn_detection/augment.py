import tensorflow as tf
from svhn_dataset import SVHN
from functools import partial
import numpy as np

def augment_py(img, bbox, width_shift=0, height_shift=0, zoom=0, rotation=0, vertical_fraction = 1.0, horizontal_fraction=1.0):

    #print(bbox)

    img_height = img.shape[0]
    img_width = img.shape[1]

    idg = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        zoom_range=zoom,
        rotation_range=rotation,
    )
    transform = idg.get_random_transform(img.shape)

    img_transformed = idg.apply_transform(img.numpy(), transform)

    rot_rad = np.deg2rad(transform['theta'])
    center_tmp = tf.convert_to_tensor([img_height/2, img_width/2, img_height/2, img_width/2], dtype=tf.double)
    centered = tf.cast(bbox, tf.double) - center_tmp
    #print(centered)
    lefttop = tf.stack([centered[:,SVHN.TOP], centered[:,SVHN.LEFT]], axis=1)
    righttop = tf.stack([centered[:,SVHN.TOP], centered[:,SVHN.RIGHT]], axis=1)
    leftbottom = tf.stack([centered[:,SVHN.BOTTOM], centered[:,SVHN.LEFT]], axis=1)
    rightbottom = tf.stack([centered[:,SVHN.BOTTOM], centered[:,SVHN.RIGHT]], axis=1)

    new_top_vals = [
        lefttop[:,1] * np.sin(rot_rad) + lefttop[:,0] * np.cos(rot_rad),
        righttop[:,1] * np.sin(rot_rad) + righttop[:,0] * np.cos(rot_rad)
    ]
    new_top_min = tf.minimum(*new_top_vals)
    new_top_max = tf.maximum(*new_top_vals)
    new_top = new_top_max - vertical_fraction * (new_top_max - new_top_min)
    new_bottom_vals = [
        leftbottom[:,1] * np.sin(rot_rad) + leftbottom[:,0] * np.cos(rot_rad),
        rightbottom[:,1] * np.sin(rot_rad) + rightbottom[:,0] * np.cos(rot_rad)
    ]
    new_bottom_min = tf.minimum(*new_bottom_vals)
    new_bottom_max = tf.maximum(*new_bottom_vals)
    new_bottom = new_bottom_min + vertical_fraction * (new_bottom_max - new_bottom_min)

    new_left_vals = [
        lefttop[:,1] * np.cos(rot_rad) - lefttop[:,0] * np.sin(rot_rad),
        leftbottom[:,1] * np.cos(rot_rad) - leftbottom[:,0] * np.sin(rot_rad)
    ]
    new_left_min = tf.minimum(*new_left_vals)
    new_left_max = tf.maximum(*new_left_vals)
    new_left = new_left_max - horizontal_fraction * (new_left_max - new_left_min)
    new_right_vals = [
        righttop[:,1] * np.cos(rot_rad) - righttop[:,0] * np.sin(rot_rad),
        rightbottom[:,1] * np.cos(rot_rad) - rightbottom[:,0] * np.sin(rot_rad)
    ]
    new_right_min = tf.minimum(*new_right_vals)
    new_right_max = tf.maximum(*new_right_vals)
    new_right = new_right_min + horizontal_fraction * (new_right_max - new_right_min)

    rotate_box = tf.stack([
        new_top,
        new_left,
        new_bottom,
        new_right,
    ], axis=1)
    #print(rotate_box)

    translate_tmp = tf.convert_to_tensor([-transform['tx'], -transform['ty'], -transform['tx'], -transform['ty']], dtype=tf.double)
    translate_bbox = tf.add(rotate_box, translate_tmp)
    #print(translate_bbox)

    zoom_tmp = tf.convert_to_tensor([transform['zx'], transform['zy'], transform['zx'], transform['zy']], dtype=tf.double)
    #print(translate_bbox / zoom_tmp)
    zoom_bbox = tf.cast(translate_bbox / zoom_tmp + center_tmp, tf.int64)
    #print(zoom_bbox)

    return img_transformed, zoom_bbox


def build_augment_map(use_rotations = True):
    def augment_map(bboxes, img):
        shift = 0.2
        rotation = 15
        zoom = 0.2
        return augment_py(img, bboxes, width_shift=shift, 
                height_shift=shift, zoom=zoom,
                rotation=rotation if use_rotations else 0, vertical_fraction=0.6,
                horizontal_fraction=0.8)
    return augment_map


def build_augment(use_rotations = True):
    augment_map = build_augment_map(use_rotations)

    @tf.function
    def augment(x):
        bboxes, image, classes = x['bboxes'], x['image'], x['classes']
        result = tf.py_function(
            augment_map,
            inp=[bboxes, image],
            Tout=[tf.int64, tf.int64]
        )
        return {
            'bboxes': tf.cast(result[1], tf.float32),
            'image': tf.cast(result[0], tf.float32),
            'classes': classes
        }

