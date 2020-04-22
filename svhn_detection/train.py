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
from data import create_data, NUM_TRAINING_SAMPLES, generate_evaluation_data, scale_input
import os
from coco_eval import CocoEvaluation




def parse_args(argv = None, skip_name = False): 
    all_argv = list(sys.argv)
    if argv is not None: all_argv.extend(argv)
    argstr = ' '.join(all_argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--image-size', default=128, type=int)
    parser.add_argument('--pyramid-levels', default=4, type=int)
    parser.add_argument('--num-scales', default=3, type=int)
    parser.add_argument('--learning-rate', default=0.16, type=float, help='0.16 in efficientdet')
    parser.add_argument('--weight-decay', default=4e-5, type=float, help='4e-5 in efficientdet')
    parser.add_argument('--momentum', default=0.9, type=float, help='0.9 in efficientdet')
    parser.add_argument('--grad-clip', default=1.0, type=float, help='not used in efficientdet')
    parser.add_argument('--score-threshold', default=0.5, type=float)
    parser.add_argument('--iou-threshold', default=0.2, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--efficientdet-filters', default=64, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--disable-gpu', action='store_true')
    parser.add_argument('--aspect-ratios-y', type=float, default=[1.4, 1.0], nargs='+') 
    parser.add_argument('--augmentation', default='none', help='One of the following: none, retina, retina-rotate, autoaugment')
    if 'JOB' in os.environ:
        parser.add_argument('--name', default=os.environ['JOB'])
    elif '--test' in all_argv or skip_name:
        parser.add_argument('--name', default='test_test')
    else:
        parser.add_argument('--name', required=True)

    if argv is not None: args = parser.parse_args(argv)
    else: args = parser.parse_args()

    assert '_' in args.name
    args.project, args.name = args.name[:args.name.index('_')], args.name[args.name.index('_') + 1:]
    if args.test:
        args.batch_size = 2

    args.aspect_ratios = [(y, round(1 / y, 1)) for y in args.aspect_ratios_y]
    delattr(args, 'aspect_ratios_y')
    return args, argstr


def output_predictions(model, dataset = 'test', filename='svhn_classification_{dataset}.txt'):
    correct = 0
    total = 0
    with open(filename.format(dataset = dataset), "w", encoding="utf-8") as out_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        dataset = getattr(SVHN(), dataset).map(SVHN.parse)
        mapped_dataset = dataset.map(scale_input(model.args.image_size))
        for x, xorig in zip(mapped_dataset.batch(1), dataset):
            predicted_bboxes, scores, predicted_classes, valid = model.predict_on_batch(x)
            num_valid = valid[0].numpy()
            predicted_bboxes = predicted_bboxes[0, :num_valid, ...].numpy() 
            predicted_classes = predicted_classes[0, :num_valid, ...].numpy()
            scores = scores[0, :num_valid, ...].numpy()

            predicted_bboxes = predicted_bboxes[scores > model.args.score_threshold]
            predicted_classes = predicted_classes[scores > model.args.score_threshold]

            transformed_bboxes = [] 
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output.append(label.astype(np.int32))
                bbox_transformed = tf.cast(tf.shape(xorig['image'])[1], tf.float32).numpy() * bbox  / float(model.args.image_size)
                transformed_bboxes.append(bbox_transformed.astype(np.int32))
                output.extend(bbox_transformed.astype(np.int32))
            print(*output, file=out_file)


            correct += utils.correct_predictions(xorig["classes"].numpy(), xorig["bboxes"].numpy(),
                    predicted_classes.astype(np.int32), transformed_bboxes)
            total += 1
    return correct / total


class RetinaTrainer:
    def __init__(self, model, anchors, dataset, val_dataset, args):
        self.model = model
        self.anchors = anchors
        self.args = args
        self.dataset = dataset
        self.val_dataset = val_dataset

        self.coco_metric = CocoEvaluation(val_dataset)


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
            class_loss = tfa.losses.sigmoid_focal_crossentropy(class_g, class_pred, from_logits=True, alpha=0.25, gamma=1.5) 
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
        class_loss = tfa.losses.sigmoid_focal_crossentropy(class_g, class_pred, from_logits=True, alpha=0.25, gamma=1.5) 
        class_loss = utils.mask_reduce_sum_over_batch(class_loss, c_mask)
        regression_loss = self._huber_loss(bbox_g, bbox_pred)
        regression_loss = utils.mask_reduce_sum_over_batch(regression_loss, r_mask)
        loss = class_loss + regression_loss

        # TODO: compute mAP
        return (loss, regression_loss, class_loss)

    @tf.function
    def predict_on_batch(self, x):
        class_pred, bbox_pred = self.model(x['image'], training=False)
        regression_pred = utils.bbox_from_fast_rcnn(self.anchors, bbox_pred) 
        regression_pred = tf.expand_dims(regression_pred, 2)
        class_pred = tf.nn.sigmoid(class_pred)
        boxes, scores, classes, valid = tf.image.combined_non_max_suppression(
            regression_pred, class_pred, 5, 5, score_threshold=0.2,#0.05,
            iou_threshold=self.args.iou_threshold, clip_boxes=False) 

        # Clip bounding boxes
        boxes = tf.clip_by_value(boxes, 0, self.args.image_size)
        return boxes, scores, classes, valid

    def predict(self, dataset = None):
        predictions = []
        if dataset is None: dataset = self.val_dataset
        dataset = dataset.batch(self.args.batch_size).prefetch(4)

        for x in dataset: 
            for boxes, scores, classes, valid_detections in zip(*map(lambda x: x.numpy(), self.predict_on_batch(x))):
                boxes, scores, classes = boxes[:valid_detections], scores[:valid_detections], classes[:valid_detections]
                predictions.append((boxes, classes, scores))
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
            log_append = dict()
            
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
            # TODO: vectorize Straka's metric
            predictions = self.predict(self.val_dataset)
            for (boxes, classes, scores), gold in zip(predictions, self.val_dataset):
                gold_classes, gold_boxes = gold['gt-class'].numpy(), gold['gt-bbox'].numpy()
                num_gt = gold['gt-length'].numpy()
                gold_classes, gold_boxes = gold_classes[:num_gt], gold_boxes[:num_gt]
                boxes, classes = boxes[scores > self.args.score_threshold], classes[scores > self.args.score_threshold]
                self.metrics['val_score'].update_state(utils.correct_predictions(gold_classes, gold_boxes, classes, boxes))
            
            # mAP metric should be implemented here. Note, that predictions
            # That are generated use transformed bb, i.e., the bb is scaled to args.image_size
            # the original dataset has different image sizes and this needs to be taken care of 
            # in the metric
            predictions = self.predict(self.val_dataset)            
            self.coco_metric.evaluate(predictions)


            # Save model every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.save()
                val_acc = output_predictions(self, 'dev')
                output_predictions(self, 'test')
                if hasattr(self, '_wandb'):
                    self._wandb.save('svhn_classification_dev.txt')
                    self._wandb.save('svhn_classification_test.txt')
                print('model saved')
                print(f'validation score: {val_acc * 100:.2f}')
                log_append = dict(saved_val_score=val_acc, **log_append)

            # Log current values
            self.log(**log_append)

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


    def log(self, **kwargs):
        values = {k: v.result().numpy() for k, v in self.metrics.items()}
        values['epoch'] = self._epoch.numpy()
        values['lr'] = self.scheduler().numpy()
        values['wd'] = self.wd_scheduler().numpy()
        values.update(kwargs)

        for key, value in self.coco_metric.get_last_stats().items():
            values[key] = value 
            
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
    anchors = utils.generate_anchors(pyramid_levels, args.image_size, 
            num_scales=args.num_scales, aspect_ratios=args.aspect_ratios)

    train_dataset, dev_dataset, _ = create_data(args.batch_size, 
            anchors, image_size = args.image_size,
            test=args.test, augmentation=args.augmentation)

    # Prepare network and trainer
    anchors_per_level = args.num_scales * len(args.aspect_ratios)
    network = efficientdet.EfficientDet(num_classes, anchors_per_level,
            input_size = args.image_size, pyramid_levels = pyramid_levels,
            filters=args.efficientdet_filters) 
    model = RetinaTrainer(network, anchors, train_dataset, dev_dataset, args)

    # Start training
    print(f'running command: {argstr}') 
    model.fit()

    # Save model
    model.save()
    print('model saved')

