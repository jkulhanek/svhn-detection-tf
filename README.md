# svhn-detection-tf
Object detection on SVHN dataset in tensorflow using efficientdet

## Getting Started
Please verify the validity of your changes before pushing by running `$ python train.py --test`.

## Experiments
TODO: add a table with results

## Tasks
Tasks should be completed in chronological order.

### Verify correct implementation of Straka's metric
Verify that predictions are computed correctly (it returns a sane value) and the val_score is computed correctly from the obtained predictions.

### Implement COCO metrics
COCO metric should validate the performance of the model during training and after the training finished. The implementation could be based on https://github.com/google/automl/blob/master/efficientdet/coco_metric.py and could used the predict function (similar to straka metric). In the fit function, the predictions are already computed. The metric could use pycocotools.

### Implement augmentations from efficientdet paper
Augmentations should be the following (if I remember correctly): translation, rotation, scale, flipY. Bounding boxes should be transformed accordingly. The correct behaviour should be verified in jupiter with bounding boxes drawn over the picture

### Experiment with different image_size
Make sure, that the smallest conv layer in efficientdet has at least 4 pixels. Increase/decrease number of efficientdet layers if needed

### Experiment with different efficientdet architecture
Try using more/less features from the efficientdet or even replace the efficientdet with a single output and map it to multi-sized anchors.

### Experiment with different learning rate/wd/momentum/grad_clip

### Compare with https://github.com/penny4860/retinanet-digit-detector

### Finetune object detection
Experiment with iouthreshold and scorethreshold to obtain maximal performance out of the network. Also finetune/replace the combined non-maximum suppression (read the docs).

### Evaluate on Straka's test set

### Finetune performance
Computing metrics could be done in better way
mAP could be implemented as a keras metric
Data pipeline could be optimized
