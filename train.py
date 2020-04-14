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
from data import create_data
import os

# TODO: finish
def predict_bounding_boxes(anchors, image_size, class_pred, regression_pred):
    regression_pred = utils.bbox_from_fast_rcnn(anchors, regression_pred)
    return tf.image.combined_non_max_suppression(
        regression_pred, class_pred, 3, 5, iou_threshold=0.5, clip_boxes=True
    ) 


def parse_args(): 
    argstr = ' '.join(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--image_size', default=128, type=int)
    parser.add_argument('--pyramid_levels', default=3, type=int)
    parser.add_argument('--num_scales', default=1, type=int)
    parser.add_argument('--learning_rate', default=0.08, type=float, help='0.16 in efficientdet')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='4e-5 in efficientdet')
    parser.add_argument('--momentum', default=0.1, type=float, help='0.9 in efficientdet')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='not used in efficientdet')
    parser.add_argument('--epochs', default=70, type=int)
    parser.add_argument('--test', action='store_true')
    if 'JOB' in os.environ:
        parser.add_argument('--name', default=os.environ['JOB'])
    elif '--test' in sys.argv:
        parser.add_argument('--name', default='test_test')
    else:
        parser.add_argument('--name', required=True)

    args = parser.parse_args()
    args.nowandb = False
    assert '_' in args.name
    args.project, args.name = args.name[:args.name.index('_')], args.name[args.name.index('_') + 1:]
    if args.test:
        args.batch_size = 2

    args.aspect_ratios = [(1.4, 0.7)]
    return args, argstr


class RetinaTrainer:
    def __init__(self, model, dataset, val_dataset, args):
        self.model = model
        self.args = args
        self.dataset = dataset \
            .batch(args.batch_size) \
            .prefetch(4)

        self.val_dataset = val_dataset \
            .batch(args.batch_size) \
            .prefetch(4)

        # Prepare training
        self._num_minibatches = self.dataset.reduce(0, lambda a,x: a + 1) # TOO SLOW
        self._huber_loss = tf.keras.losses.Huber(reduction = tf.losses.Reduction.NONE)
        self._epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
        self._epoch_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.scheduler = utils.WarmStartCosineDecay(0.16, args.epochs, self._num_minibatches, self._epoch, self._epoch_step)
        self.wd_scheduler = utils.WarmStartCosineDecay(0.16, args.epochs, self._num_minibatches, self._epoch, self._epoch_step)
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
        }

        if not args.test:
            # Use wandb
            import wandb
            wandb.init(project=args.project, name=args.name)
            wandb.config.update(args)
            self._wandb = wandb


    @tf.function
    def train_on_batch(self, x, y):
        with tf.GradientTape() as tp:
            class_pred, bbox_pred = self.model(x)
            class_g, bbox_g, c_mask, r_mask = y['class'], y['bbox'], y['class_mask'], y['regression_mask']
            class_loss = tfa.losses.sigmoid_focal_crossentropy(class_g, class_pred, from_logits=True) 
            class_loss = utils.mask_reduce_sum_over_batch(class_loss, c_mask)
            regression_loss = self._huber_loss(bbox_g, bbox_pred)
            regression_loss = utils.mask_reduce_sum_over_batch(regression_loss, r_mask)
            loss = class_loss + regression_loss
        grads = tp.gradient(loss, self.model.trainable_variables)
        capped_grads = (tf.clip_by_value(grad, -1., 1.) for grad in grads)
        self.optimizer.apply_gradients(zip(capped_grads, self.model.trainable_variables))
        return (loss, regression_loss, class_loss)

    @tf.function
    def evaluate_on_batch(self, x, y):
        class_pred, bbox_pred = self.model(x)
        class_g, bbox_g, c_mask, r_mask = y['class'], y['bbox'], y['class_mask'], y['regression_mask']
        class_loss = tfa.losses.sigmoid_focal_crossentropy(class_g, class_pred, from_logits=True) 
        class_loss = utils.mask_reduce_sum_over_batch(class_loss, c_mask)
        regression_loss = self._huber_loss(bbox_g, bbox_pred)
        regression_loss = utils.mask_reduce_sum_over_batch(regression_loss, r_mask)
        loss = class_loss + regression_loss

        # TODO: compute mAP
        return (loss, regression_loss, class_loss)

    def fit(self): 
        for epoch in range(self.args.epochs):
            self._epoch.assign(epoch)
            
            # Reset metrics
            for m in self.metrics.values(): m.reset_states()

            # Train on train dataset
            for epoch_step, (x, y) in enumerate(self.dataset):
                self._epoch_step.assign(epoch_step)
                loss, regression_loss, class_loss = self.train_on_batch(x, y)
                self.metrics['loss'].update_state(loss)
                self.metrics['regression_loss'].update_state(regression_loss)
                self.metrics['class_loss'].update_state(class_loss)

            # Run validation
            for x, y in self.val_dataset:
                loss, regression_loss, class_loss = self.evaluate_on_batch(x, y)
                self.metrics['val_loss'].update_state(loss)
                self.metrics['val_regression_loss'].update_state(regression_loss)
                self.metrics['val_class_loss'].update_state(class_loss)

            # Save model every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.save()
                print('model saved')

            # Log current values
            self.log()

    def log(self):
        values = {k: v.result().numpy() for k, v in self.metrics.items()}
        values['epoch'] = self._epoch.numpy()
        values['lr'] = self.scheduler().numpy()
        values['wd'] = self.wd_scheduler().numpy()
        if hasattr(self, '_wandb'):
            # We will use wandb
            self._wandb.log(values, step=values['epoch'])
        console_metrics = ['epoch: {epoch}', 'loss: {loss:.4f}', 'val_loss: {val_loss:.4f}',
                'val_class_loss: {val_class_loss:.4f}', 'val_reg_loss: {val_regression_loss:.4f}']
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
            test=args.test, evaluation=False)

    # Prepare network and trainer
    anchors_per_level = args.num_scales * len(args.aspect_ratios)
    network = efficientdet.EfficientDet(num_classes, anchors_per_level,
            input_size = args.image_size, pyramid_levels = pyramid_levels) 
    model = RetinaTrainer(network, train_dataset, dev_dataset, args)

    # Start training
    print(f'running command: {argstr}') 
    model.fit()

    # Save model
    model.save()
    print('model saved')

