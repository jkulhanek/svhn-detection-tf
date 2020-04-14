import pytest
import numpy as np
import tensorflow as tf
from utils import *

@pytest.mark.parametrize('anchor,bbox,fast_rcnn', [
            [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0]],
            [[0, 0, 10, 10], [5, 0, 15, 10], [.5, 0, 0, 0]],
            [[0, 0, 10, 10], [0, 5, 10, 15], [0, .5, 0, 0]],
            [[0, 0, 10, 10], [0, 0, 20, 20], [.5, .5, np.log(2), np.log(2)]]])
def test_bbox_to_from_fast_rcnn(anchor, bbox, fast_rcnn):
    np.testing.assert_almost_equal(bbox_to_fast_rcnn(tf.convert_to_tensor(anchor,tf.float32), tf.convert_to_tensor(bbox, tf.float32)), fast_rcnn, decimal=3)
    np.testing.assert_almost_equal(bbox_from_fast_rcnn(tf.convert_to_tensor(anchor, tf.float32), tf.convert_to_tensor(fast_rcnn, tf.float32)), bbox, decimal=3)

@pytest.mark.parametrize('gold_classes,gold_bboxes,anchor_classes,anchor_bboxes,iou', [
    [[1], [[14, 14, 16, 16]], [0, 0, 0, 2], [[0, 0, 0, 0]] * 3 + [[0, 0, np.log(1/5), np.log(1/5)]], 0.5],
    [[2], [[0, 0, 20, 20]], [3, 0, 0, 0], [[.5, .5, np.log(2), np.log(2)]] + [[0, 0, 0, 0]] * 3, 0.26],
    [[2], [[0, 0, 20, 20]], [3, 3, 3, 3], [[y, x, np.log(2), np.log(2)] for y in [.5, -.5] for x in [.5, -.5]], 0.24],
])
def test_bboxes_training(gold_classes, gold_bboxes, anchor_classes, anchor_bboxes, iou):
    anchors = [[0, 0, 10, 10], [0, 10, 10, 20], [10, 0, 20, 10], [10, 10, 20, 20]]
    computed_classes, computed_bboxes, _ = bboxes_training(tf.convert_to_tensor(anchors, tf.float32), tf.convert_to_tensor(gold_classes, tf.int32), tf.convert_to_tensor(gold_bboxes, tf.float32), iou, iou, True)
    np.testing.assert_almost_equal(computed_classes.numpy(), anchor_classes, decimal=3)
    np.testing.assert_almost_equal(computed_bboxes.numpy(), anchor_bboxes, decimal=3)

def test_recodex_training():
    anchors = [[0, 0, 5, 5], [0, 5, 5, 10], [0, 10, 5, 15], [0, 15, 5, 20], [0, 20, 5, 25], [5, 0, 10, 5], [5, 5, 10, 10], [5, 10, 10, 15], [5, 15, 10, 20], [5, 20, 10, 25], [10, 0, 15, 5], [10, 5, 15, 10], [10, 10, 15, 15], [10, 15, 15, 20], [10, 20, 15, 25], [15, 0, 20, 5], [15, 5, 20, 10], [15, 10, 20, 15], [15, 15, 20, 20], [15, 20, 20, 25], [20, 0, 25, 5], [20, 5, 25, 10], [20, 10, 25, 15], [20, 15, 25, 20], [20, 20, 25, 25]]
    gold_classes = [7, 6, 3, 4, 5, 9, 9, 2]
    gold_bboxes = [[20, 6, 30, 9], [10, 10, 18, 15], [7, 23, 10, 29], [1, 23, 13, 37], [1, 20, 2, 32], [21, 11, 30, 12], [15, 14, 29, 28], [11, 22, 15, 31]]
    iou = 0.13716033017599819
    anchor_classes = [ 0,  0,  0,  0,  6,  0,  0,  0,  0,  4,  0,  0,  7,  0,  3,  0,  0, 7, 10,  0,  0,  8, 10,  0,  0]
    computed_classes, computed_bboxes,_ = bboxes_training(tf.convert_to_tensor(anchors,tf.float32), tf.convert_to_tensor(gold_classes, tf.int32), tf.convert_to_tensor(gold_bboxes, tf.float32), iou, None, True)
    np.testing.assert_almost_equal(computed_classes.numpy(), anchor_classes, decimal=3)
    #np.testing.assert_almost_equal(computed_bboxes, anchor_bboxes, decimal=3)

