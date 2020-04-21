from data import create_data
from svhn_dataset import SVHN
from train import RetinaTrainer, parse_args, output_predictions
import efficientdet
import utils
from coco_eval import CocoEvaluation


if __name__ == '__main__':
    args, argstr = parse_args(skip_name = True)

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
            input_size = args.image_size, pyramid_levels = pyramid_levels) 
    model = RetinaTrainer(network, anchors, train_dataset, dev_dataset, args)

    # Load weights
    model.model.load_weights('model.h5')
    
    coco_metric = CocoEvaluation(dev_dataset)
    predictions = model.predict(dev_dataset)
    coco_metric.evaluate(predictions)

    # Export dev and test predictions
    dev_acc = output_predictions(model, 'dev') 
    print("Score on dev: {:.2f}".format(100 * dev_acc))

    output_predictions(model, 'test')
