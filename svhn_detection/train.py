import numpy as np
import os
import wandb
import sys
import argparse
import tensorflow as tf
import tensorflow_addons as tfa
import efficientdet
import numpy as np
from svhn_dataset import SVHN
import utils
import utils
from functools import partial
from data import create_data, NUM_TRAINING_SAMPLES
import os



def parse_args(argv = []): 
    all_argv = list(argv) + sys.argv
    argstr = ' '.join(all_argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--image_size', default=128, type=int)
    parser.add_argument('--pyramid_levels', default=4, type=int)
    parser.add_argument('--num_scales', default=1, type=int)
    parser.add_argument('--learning_rate', default=0.16, type=float, help='0.16 in efficientdet')
    parser.add_argument('--weight_decay', default=4e-5, type=float, help='4e-5 in efficientdet')
    parser.add_argument('--momentum', default=0.9, type=float, help='0.9 in efficientdet')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='not used in efficientdet')
    parser.add_argument('--epochs', default=70, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--disable_gpu', action='store_true')
    parser.add_argument('--augmentation', default='none', help='One of the following: none, retina, retina-rotate, autoaugment')
    if 'JOB' in os.environ:
        parser.add_argument('--name', default=os.environ['JOB'])
    elif '--test' in all_argv:
        parser.add_argument('--name', default='test_test')
    else:
        parser.add_argument('--name', required=True)

    if argv is not None: args = parser.parse_args(argv)
    else: args = parser.parse_args()

    assert '_' in args.name
    args.project, args.name = args.name[:args.name.index('_')], args.name[args.name.index('_') + 1:]
    if args.test:
        args.batch_size = 2

    args.aspect_ratios = [(1.4, 0.7)]
    return args, argstr


class RetinaTrainer:
    def __init__(self, model, anchors, dataset, val_dataset, args):
        self.model = model
        self.anchors = anchors
        self.args = args
        self.dataset = dataset

        self.eval_dataset = val_dataset
        self.val_dataset = val_dataset


        # Prepare training
        self._num_minibatches = self.dataset.reduce(0, lambda a,x: a + 1)
        self._huber_loss = tf.keras.losses.Huber(reduction = tf.losses.Reduction.NONE)
        self._epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
        self._epoch_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self._grad_clip = args.grad_clip
        self.scheduler = utils.WarmStartCosineDecay(args.learning_rate, args.epochs, self._num_minibatches, self._epoch, self._epoch_step)
        self.wd_scheduler = utils.WarmStartCosineDecay(args.weight_decay, args.epochs, self._num_minibatches, self._epoch, self._epoch_step)
        self.optimizer = tfa.optimizers.SGDW(
                self.wd_scheduler,
                learning_rate=self.scheduler,
                momentum=args.momentum,
                nesterov=True) 
        self.metrics = {
            'loss': tf.keras.metrics.Mean(),
            'class_loss': tf.keras.metrics.Mean(),
            'regression_loss': tf.keras.metrics.Mean(),
            'val_loss': tf.keras.metrics.Mean(),
            'val_class_loss': tf.keras.metrics.Mean(),
            'val_regression_loss': tf.keras.metrics.Mean(),
            'val_score': tf.keras.metrics.Mean(),
        }

    def _start_wandb(self): 
        if not self.args.test:
            # Use wandb
            import wandb
            wandb.init(entity='kulhanek', 
                    anonymous='allow',
                    project=self.args.project,
                    name=self.args.name)
            wandb.config.update(self.args)
            self._wandb = wandb

    @tf.function
    def train_on_batch(self, x):
        with tf.GradientTape() as tp:
            class_pred, bbox_pred = self.model(x['image'], training=True)
            class_g, bbox_g, c_mask, r_mask = x['class'], x['bbox'], x['class_mask'], x['regression_mask']
            class_loss = tfa.losses.sigmoid_focal_crossentropy(class_g, class_pred, from_logits=True) 
            class_loss = utils.mask_reduce_sum_over_batch(class_loss, c_mask)
            regression_loss = self._huber_loss(bbox_g, bbox_pred)
            regression_loss = utils.mask_reduce_sum_over_batch(regression_loss, r_mask)
            loss = class_loss + regression_loss
        grads = tp.gradient(loss, self.model.trainable_variables)
        capped_grads, gradient_norm = tf.clip_by_global_norm(grads, self._grad_clip)
        self.optimizer.apply_gradients(zip(capped_grads, self.model.trainable_variables))
        return (loss, regression_loss, class_loss)

    @tf.function
    def evaluate_on_batch(self, x):
        class_pred, bbox_pred = self.model(x['image'], training=False)
        class_g, bbox_g, c_mask, r_mask = x['class'], x['bbox'], x['class_mask'], x['regression_mask']
        class_loss = tfa.losses.sigmoid_focal_crossentropy(class_g, class_pred, from_logits=True) 
        class_loss = utils.mask_reduce_sum_over_batch(class_loss, c_mask)
        regression_loss = self._huber_loss(bbox_g, bbox_pred)
        regression_loss = utils.mask_reduce_sum_over_batch(regression_loss, r_mask)
        loss = class_loss + regression_loss

        # TODO: compute mAP
        return (loss, regression_loss, class_loss)

    @tf.function
    def predict_on_batch(self, x, score_threshold = 0.05):
        class_pred, bbox_pred = self.model(x['image'], training=False)
        regression_pred = utils.bbox_from_fast_rcnn(self.anchors, bbox_pred) 
        regression_pred = tf.expand_dims(regression_pred, 2)
        class_pred = tf.nn.sigmoid(class_pred)
        boxes, scores, classes, valid = tf.image.combined_non_max_suppression(
            regression_pred, class_pred, 3, 5, score_threshold=score_threshold,
            iou_threshold=0.5, clip_boxes=False) 

        # Clip bounding boxes
        boxes = tf.clip_by_value(boxes, 0, self.args.image_size)
        return boxes, scores, classes, valid

    def predict(self, dataset = None):
        predictions = []
        if dataset is None: dataset = self.val_dataset
        dataset = dataset.batch(self.args.batch_size).prefetch(4)

        for x in dataset: 
            for boxes, scores, classes, valid_detections in zip(*map(lambda x: x.numpy(), self.predict_on_batch(x))):
                predictions.append((boxes[:valid_detections], classes[:valid_detections], scores[:valid_detections]))
        return predictions


    def fit(self): 
        self._start_wandb()
        dataset = self.dataset.shuffle(3000) \
            .batch(args.batch_size) \
            .prefetch(4)

        val_dataset = self.val_dataset \
            .batch(args.batch_size) \
            .prefetch(4)
            
        for epoch in range(self.args.epochs):
            self._epoch.assign(epoch)
            
            # Reset metrics
            for m in self.metrics.values(): m.reset_states()

            # Train on train dataset
            for epoch_step, x in enumerate(dataset):
                self._epoch_step.assign(epoch_step)
                loss, regression_loss, class_loss = self.train_on_batch(x)
                self.metrics['loss'].update_state(loss)
                self.metrics['regression_loss'].update_state(regression_loss)
                self.metrics['class_loss'].update_state(class_loss)

            # Run validation
            for x in val_dataset:
                loss, regression_loss, class_loss = self.evaluate_on_batch(x)
                self.metrics['val_loss'].update_state(loss)
                self.metrics['val_regression_loss'].update_state(regression_loss)
                self.metrics['val_class_loss'].update_state(class_loss)

            # Compute straka's metric
            #predictions = self.predict(self.val_dataset)
            # for (boxes, classes, scores), gold in zip(predictions, self.val_dataset):
            #     gold_classes, gold_boxes = gold['gt-class'].numpy(), gold['gt-bbox'].numpy()
            #     num_gt = gold['gt-length'].numpy()
            #    gold_classes, gold_boxes = gold_classes[:num_gt], gold_boxes[:num_gt]
            #     self.metrics['val_score'].update_state(utils.correct_predictions(gold_boxes, gold_classes, classes, boxes))

            # Save model every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.save()
                print('model saved')

            # Log current values
            self.log()

    def evaluate(self, dataset = None):
        if dataset is None: dataset = self.val_dataset
        dataset = dataset \
            .batch(self.args.batch_size) \
            .prefetch(4)

        # Reset metrics
        for m in self.metrics.values(): m.reset_states()

        # Run validation
        for x in dataset:
            loss, regression_loss, class_loss = self.evaluate_on_batch(x)
            self.metrics['val_loss'].update_state(loss)
            self.metrics['val_regression_loss'].update_state(regression_loss)
            self.metrics['val_class_loss'].update_state(class_loss)
        return dict(
            loss = self.metrics.get('val_loss').result().numpy(),
            regression_loss = self.metrics.get('val_regression_loss').result().numpy(),
            class_loss = self.metrics.get('val_class_loss').result().numpy(),
        )


    def log(self):
        values = {k: v.result().numpy() for k, v in self.metrics.items()}
        values['epoch'] = self._epoch.numpy()
        values['lr'] = self.scheduler().numpy()
        values['wd'] = self.wd_scheduler().numpy()
        if hasattr(self, '_wandb'):
            # We will use wandb
            self._wandb.log(values, step=values['epoch'])
        console_metrics = ['epoch: {epoch}', 'loss: {loss:.4f}', 'val_loss: {val_loss:.4f}',
                'val_class_loss: {val_class_loss:.4f}', 'val_reg_loss: {val_regression_loss:.4f}',
                'val_score: {val_score:.4f}']
        print(', '.join(console_metrics).format(**values)) 

    def save(self, filename = 'model.h5'):
        self.model.save(filename)
        if hasattr(self, '_wandb'):
            self._wandb.save(filename)


if __name__ == '__main__': 
    args, argstr = parse_args()

    # Prepare data
    num_classes = SVHN.LABELS
    pyramid_levels = args.pyramid_levels
    smallest_stride = 2**(6 - pyramid_levels)
    anchors = utils.generate_anchors(pyramid_levels, args.image_size, 
            first_feature_scale=smallest_stride, anchor_scale=float(smallest_stride),
            num_scales=args.num_scales, aspect_ratios=args.aspect_ratios)

    train_dataset, dev_dataset, _ = create_data(args.batch_size, 
            anchors, image_size = args.image_size,
            test=args.test, augmentation=args.augmentation)

    # Prepare network and trainer
    anchors_per_level = args.num_scales * len(args.aspect_ratios)
    network = efficientdet.EfficientDet(num_classes, anchors_per_level,
            input_size = args.image_size, pyramid_levels = pyramid_levels) 
    model = RetinaTrainer(network, anchors, train_dataset, dev_dataset, args)

    # Start training
    print(f'running command: {argstr}') 
    model.fit()

    # Save model
    model.save()
    print('model saved')

