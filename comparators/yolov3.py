import numpy as np

from helpers.yolov3_extractor import YOLOV3Extractor
from evaluators.object_detection_yolov3 import YOLOV3ObjectDetectionEvaluator

class YOLOV3Comparator:

    def __init__(self, evaluation, comparisons):
        self.evaluation = evaluation
        self.comparisons = comparisons
        pass

    def update(self, model_name, model, base_run, opt_run, current_pass="all"):

        extractor = YOLOV3Extractor()

        metrics_5 = []
        metrics_7 = []
        metrics_9 = []
        evaluator_5 = YOLOV3ObjectDetectionEvaluator(iou_threshold=0.5)
        evaluator_7 = YOLOV3ObjectDetectionEvaluator(iou_threshold=0.75)
        evaluator_9 = YOLOV3ObjectDetectionEvaluator(iou_threshold=0.9)

        output = [node.name for node in model.graph.output]

        for img_key in base_run.keys():
            base_image = base_run[img_key]
            opt_image = opt_run[img_key]
            base_data = extractor.extract_yolo_outputs(base_image)
            base_bboxes = base_data[0]
            base_labels = base_data[1]
            base_scores = base_data[2]

            opt_data = extractor.extract_yolo_outputs(opt_image)
            opt_bboxes = opt_data[0]
            opt_labels = opt_data[1]
            opt_scores = opt_data[2]

            # Skip undetected classes.
            if (base_labels == [] and opt_labels == []) or (base_bboxes == [] and opt_bboxes == []):
                continue

            metric_5 = evaluator_5.compute_metrics(base_bboxes, base_labels, base_scores, opt_bboxes, opt_labels, opt_scores)
            metrics_5.append(metric_5)

            metric_7 = evaluator_7.compute_metrics(base_bboxes, base_labels, base_scores, opt_bboxes, opt_labels, opt_scores)
            metrics_7.append(metric_7)

            metric_9 = evaluator_9.compute_metrics(base_bboxes, base_labels, base_scores, opt_bboxes, opt_labels, opt_scores)
            metrics_9.append(metric_9)

        
        self.comparisons[model_name][current_pass][output[0]] = {
            "metrics_0_5": {
                "avg_f1": np.mean([o["F1"] for o in metrics_5]) * 100,
                "avg_prec": np.mean([o["precision"] for o in metrics_5]) * 100,
                "avg_recall": np.mean([o["recall"] for o in metrics_5]) * 100,
                "avg_mAP": np.median([o["mAP"] for o in metrics_5]) * 100,
                "avg_IoU": np.mean([o["IoU"] for o in metrics_5]) * 100,
            },
            "metrics_0_7": {
                "avg_f1": np.mean([o["F1"] for o in metrics_7]) * 100,
                "avg_prec": np.mean([o["precision"] for o in metrics_7]) * 100,
                "avg_recall": np.mean([o["recall"] for o in metrics_7]) * 100,
                "avg_mAP": np.mean([o["mAP"] for o in metrics_7]) * 100,
                "avg_IoU": np.mean([o["IoU"] for o in metrics_7]) * 100,
            },
            "metrics_0_9": {
                "avg_f1": np.mean([o["F1"] for o in metrics_9]) * 100,
                "avg_prec": np.mean([o["precision"] for o in metrics_9]) * 100,
                "avg_recall": np.mean([o["recall"] for o in metrics_9]) * 100,
                "avg_mAP": np.mean([o["mAP"] for o in metrics_9]) * 100,
                "avg_IoU": np.mean([o["IoU"] for o in metrics_9]) * 100,
            }
        }


        cmp_object = {}
        if (self.evaluation["percentage_dissimilar1"][1] != -1):
            cmp_object["first"] = self.evaluation["percentage_dissimilar1"][1]
        if (self.evaluation["percentage_dissimilar5"][1] != -1):
            cmp_object["top5"] = self.evaluation["percentage_dissimilar5"][1]
        cmp_object["topK"] = self.evaluation["percentage_dissimilar"][1]
        # Labels
        self.comparisons[model_name][current_pass][output[1]] = cmp_object
        