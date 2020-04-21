from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from svhn_dataset import SVHN


class CocoEvaluation():
    def __init__(self, golden_dataset):
        self.golden_coco = self.create_coco_from_dataset(golden_dataset)

    def create_annotations(self, dataset):
        annotations = []
        self.image_id = 0
        object_id = 1
        for gold in dataset:
            gold_classes, gold_boxes = gold['gt-class'].numpy(), gold['gt-bbox'].numpy()
            num_gt = gold['gt-length'].numpy()
            gold_classes, gold_boxes = gold_classes[:num_gt], gold_boxes[:num_gt]
            
            for gold_class, gold_box in zip(gold_classes, gold_boxes):
                x, y, width, height = gold_box[SVHN.LEFT], gold_box[SVHN.TOP], gold_box[SVHN.RIGHT] - gold_box[SVHN.LEFT],  gold_box[SVHN.BOTTOM] - gold_box[SVHN.TOP] 
                annotations.append({
                    'id': object_id,
                    'area' : (gold_box[SVHN.RIGHT] - gold_box[SVHN.LEFT]) * (gold_box[SVHN.BOTTOM] -  gold_box[SVHN.TOP]),
                    'bbox': [x, y, width, height],
                    'category_id': gold_class,
                    'image_id': self.image_id,
                    'iscrowd': 0,
                })
                object_id += 1
            self.image_id += 1
        return annotations

    def create_images(self):
        images = []
        for i in range(self.image_id):
            images.append({
                "id": i
            })
        return images

    def create_categories(self):
        categories = []
        for i in range(10):
            categories.append({
                "id": i,    
            })
        return categories

    def create_coco_from_dataset(self, dataset):                
        coco = COCO()
        coco_dataset = {}
        coco_dataset['annotations'] = self.create_annotations(dataset)
        coco_dataset['images'] = self.create_images()
        coco_dataset['categories'] = self.create_categories()
        coco.dataset = coco_dataset
        coco.createIndex()

        return coco

    def create_coco_from_predictions(self, predictions):
        # boxes, scores, classes, valid
        result = []
        for (boxes, classes, scores), img_id in zip(predictions, range(self.image_id)):
            for box, det_class, score in zip(boxes, classes, scores):
                result.append({
                    'image_id': img_id,
                    'category_id': int(det_class),
                    'bbox': [box[SVHN.LEFT], box[SVHN.TOP], box[SVHN.RIGHT] - box[SVHN.LEFT],  box[SVHN.BOTTOM] - box[SVHN.TOP] ],
                    'area': (box[SVHN.RIGHT] - box[SVHN.LEFT]) * (box[SVHN.BOTTOM] -  box[SVHN.TOP]),
                    'iscrowd': 0,
                    'score': score
                })
        return result
                

    
    def evaluate(self, predictions):
        detections = self.create_coco_from_predictions(predictions)
        if len(detections) == 0:
            return 
            
        coco_dt = self.golden_coco.loadRes(detections)
        coco_eval = COCOeval(self.golden_coco, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


        
    