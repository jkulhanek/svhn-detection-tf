#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from svhn_dataset import SVHN

def bbox_area(a):
    return tf.maximum(tf.zeros_like(a[...,2]), a[...,2] - a[...,0]) * tf.maximum(tf.zeros_like(a[...,2]), a[...,3] - a[...,1]) 


def bbox_iou(a, b):
    """ Compute IoU for two bboxes a, b.

    Each bbox is parametrized as a four-tuple (top, left, bottom, right).
    """
    a = tf.expand_dims(a, -2)
    b = tf.expand_dims(b, -3)
    intersection = tf.stack([
        tf.maximum(a[...,0], b[...,0]),
        tf.maximum(a[...,1], b[...,1]),
        tf.minimum(a[...,2], b[...,2]),
        tf.minimum(a[...,3], b[...,3]),
    ], -1)
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    area_intersection = bbox_area(intersection)
    area_union = area_a + area_b - area_intersection
    filter_expr = area_intersection > 0
    area_intersection = tf.where(filter_expr, area_intersection / area_union, area_intersection)
    return area_intersection


def np_bbox_area(a):
    return max(0, a[SVHN.BOTTOM] - a[SVHN.TOP]) * max(0, a[SVHN.RIGHT] - a[SVHN.LEFT])

def np_bbox_iou(a, b):
    """ Compute IoU for two bboxes a, b.
    Each bbox is parametrized as a four-tuple (top, left, bottom, right).
    """
    intersection = [
        max(a[SVHN.TOP], b[SVHN.TOP]),
        max(a[SVHN.LEFT], b[SVHN.LEFT]),
        min(a[SVHN.BOTTOM], b[SVHN.BOTTOM]),
        min(a[SVHN.RIGHT], b[SVHN.RIGHT]),
    ]
    if intersection[SVHN.RIGHT] <= intersection[SVHN.LEFT] or intersection[SVHN.BOTTOM] <= intersection[SVHN.TOP]:
        return 0
    return np_bbox_area(intersection) / float(np_bbox_area(a) + np_bbox_area(b) - np_bbox_area(intersection))


def bbox_to_fast_rcnn(anchor, bbox):
    """ Convert `bbox` to a Fast-R-CNN-like representation relative to `anchor`.

    The `anchor` and `bbox` are four-tuples (top, left, bottom, right);
    you can use SVNH.{TOP, LEFT, BOTTOM, RIGHT} as indices.

    The resulting representation is a four-tuple with:
    - (bbox_y_center - anchor_y_center) / anchor_height
    - (bbox_x_center - anchor_x_center) / anchor_width
    - log(bbox_height / anchor_height)
    - log(bbox_width / anchor_width)
    """
    bbox_height = bbox[...,SVHN.BOTTOM] - bbox[...,SVHN.TOP]
    bbox_width = bbox[...,SVHN.RIGHT] - bbox[...,SVHN.LEFT]
    anchor_height = anchor[...,SVHN.BOTTOM] - anchor[...,SVHN.TOP]
    anchor_width = anchor[...,SVHN.RIGHT] - anchor[...,SVHN.LEFT]
    bbox_y_center = 0.5 * (bbox_height) + bbox[...,SVHN.TOP]
    bbox_x_center = 0.5 * (bbox_width) + bbox[...,SVHN.LEFT]
    anchor_y_center = 0.5 * (anchor_height) + anchor[...,SVHN.TOP]
    anchor_x_center = 0.5 * (anchor_width) + anchor[...,SVHN.LEFT]
    return tf.stack([
        (bbox_y_center - anchor_y_center) / anchor_height,
        (bbox_x_center - anchor_x_center) / anchor_width,
        tf.math.log(bbox_height / anchor_height),
        tf.math.log(bbox_width / anchor_width),
    ], -1)

def bbox_from_fast_rcnn(anchor, fast_rcnn):
    """ Convert Fast-R-CNN-like representation relative to `anchor` to a `bbox`."""
    anchor_height = anchor[...,SVHN.BOTTOM] - anchor[...,SVHN.TOP]
    anchor_width = anchor[...,SVHN.RIGHT] - anchor[...,SVHN.LEFT]
    anchor_y_center = 0.5 * (anchor_height) + anchor[...,SVHN.TOP]
    anchor_x_center = 0.5 * (anchor_width) + anchor[...,SVHN.LEFT]

    center_y, center_x, height, width = fast_rcnn[...,0], fast_rcnn[...,1],fast_rcnn[...,2],fast_rcnn[...,3]
    bbox_height = tf.exp(height) * anchor_height
    bbox_width = tf.exp(width) * anchor_width
    bbox_y_center = center_y * anchor_height + anchor_y_center
    bbox_x_center = center_x * anchor_width + anchor_x_center
    return tf.stack([
        bbox_y_center - bbox_height * 0.5,
        bbox_x_center - bbox_width * 0.5,
        bbox_y_center + bbox_height * 0.5,
        bbox_x_center + bbox_width * 0.5,
    ], -1)

def compute_matches(iou_table, iou_threshold, background_iou_threshold, force_gold_match):
    """ 
    Itfut matrix has shape [gold_boxes, anchors]
    Returns: [final_matches, final_mask, anchor_mask],
    where anchor_mask is one for background and foregroun objects and 0 for objects in the critical interval
    between background iou threshold and iou threshold
    """
    matches = tf.argmax(iou_table, 0) 
    matched_vals = tf.reduce_max(iou_table, 0)
    mask = matched_vals >= iou_threshold
    if background_iou_threshold is None:
        anchor_mask = tf.ones_like(mask)
    else:
        anchor_mask = tf.logical_or(mask, matched_vals < background_iou_threshold)

    if force_gold_match:
        force_match_column_ids = tf.argmax(iou_table, 1)
        force_match_column_indicators = tf.one_hot(force_match_column_ids, depth = tf.shape(iou_table)[1])
        force_match_row_ids = tf.argmax(force_match_column_indicators, 0)
        force_match_column_mask = tf.cast(tf.reduce_max(force_match_column_indicators, 0), tf.bool)
        matches = tf.where(force_match_column_mask, force_match_row_ids, matches)
        mask = tf.logical_or(mask, force_match_column_mask)
        anchor_mask = tf.logical_or(anchor_mask, force_match_column_mask)

    return matches, mask, anchor_mask


def bboxes_training(anchors, gold_classes, gold_bboxes, iou_threshold = 0.5, background_iou_threshold = 0.4, force_gold_match = False):
    """ Compute training data for object detection.

    Arguments:
    - `anchors` is an array of four-tuples (top, left, bottom, right)
    - `gold_classes` is an array of zero-based classes of the gold objects
    - `gold_bboxes` is an array of four-tuples (top, left, bottom, right)
      of the gold objects
    - `iou_threshold` is a given threshold

    Returns:
    - `anchor_classes` contains for every anchor either 0 for background
      (if no gold object is assigned) or `1 + gold_class` if a gold object
      with `gold_class` as assigned to it
    - `anchor_bboxes` contains for every anchor a four-tuple
      `(center_y, center_x, height, width)` representing the gold bbox of
      a chosen object using parametrization of Fast R-CNN; zeros if not
      gold object was assigned to the anchor
    """
    num_anchors = tf.shape(anchors)[0]
    anchor_classes = tf.zeros((num_anchors,), tf.int32)
    assigned_bboxes = tf.zeros((num_anchors, 4), tf.float32)

    iou_table = bbox_iou(gold_bboxes, anchors)
    matches, mask, anchor_mask = compute_matches(iou_table, iou_threshold, background_iou_threshold, force_gold_match)
    anchor_classes = tf.where(mask, tf.gather_nd(gold_classes, matches[:,tf.newaxis]) + 1, anchor_classes)
    anchor_bboxes = tf.where(mask[:, tf.newaxis], 
        tf.gather_nd(gold_bboxes, matches[:, tf.newaxis]), 
        tf.zeros((num_anchors, 4), tf.float32))
    return anchor_classes, anchor_bboxes, anchor_mask



def generate_anchors(pyramid_levels, image_size, first_feature_scale=4, anchor_scale=4.0, aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)], num_scales=3): 
    boxes_all = []
    for s in range(pyramid_levels):
        boxes_level = []
        for octave in range(num_scales):
            for aspect_h, aspect_w in aspect_ratios:
                scale = 2 ** (octave / num_scales)
                stride = first_feature_scale * 2 ** s
                base_anchor_size = anchor_scale * stride * scale
                anchor_size_x = base_anchor_size * aspect_w / 2.0
                anchor_size_y = base_anchor_size * aspect_h / 2.0
            
                x = np.arange(stride / 2, image_size, stride)
                y = np.arange(stride / 2, image_size, stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                boxes = np.vstack((yv - anchor_size_y, xv - anchor_size_x,
                    yv + anchor_size_y, xv + anchor_size_x))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape(-1, 4))
    return np.vstack(boxes_all) 


def WarmStartCosineDecay(initial_learning_rate, num_epochs, num_batches, epoch, epoch_step): 
    cosine_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate, num_epochs - 1)
    def compute():
        minibatch_progress = tf.cast(epoch_step + 1, tf.float32) / float(num_batches)
        first_epoch = tf.cast(epoch == 0, tf.float32)
        cosine_decay = cosine_schedule(epoch - 1)
        first_epoch_schedule = minibatch_progress * initial_learning_rate
        return first_epoch * first_epoch_schedule + (1 - first_epoch) * cosine_decay
    return compute 


@tf.function
def mask_reduce_sum_over_batch(values, mask):
    batch_size = tf.cast(tf.shape(values)[0], tf.float32)
    if len(tf.shape(values)) == 3:
        mask = tf.expand_dims(mask, -1)
    masked_values = tf.where(mask, values, 0.0)
    return tf.reduce_sum(masked_values) / batch_size


def correct_predictions(gold_classes, gold_bboxes, predicted_classes, predicted_bboxes, iou_threshold=0.5):
    if len(gold_classes) != len(predicted_classes):
        return False
    
    used = [False] * len(gold_classes)
    for cls, bbox in zip(predicted_classes, predicted_bboxes):
        best = None
        for i in range(len(gold_classes)):
            if used[i] or gold_classes[i] != cls:
                continue
            iou = np_bbox_iou(bbox, gold_bboxes[i])
            if iou >= iou_threshold and (best is None or iou > best_iou):
                best, best_iou = i, iou
        if best is None:
            return False
        used[best] = True
    return True



