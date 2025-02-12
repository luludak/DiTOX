import numpy as np
from collections import defaultdict

class SSDObjectDetectionEvaluator:
    def __init__(self, iou_threshold=0.5):
        """
        Initialize the evaluator with IoU threshold.
        """
        self.iou_threshold = iou_threshold

    def compute_iou(self, box1, box2):
        """
        Compute IoU (Intersection over Union) between two bounding boxes.
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        inter_x1 = max(x1, x1g)
        inter_y1 = max(y1, y1g)
        inter_x2 = min(x2, x2g)
        inter_y2 = min(y2, y2g)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def compute_f1(self, bboxes_A, labels_A, scores_A, bboxes_B, labels_B, scores_B):
        """
        Compute F1 by comparing model B (optimized) predictions against model A (original) as pseudo-ground truth.
        
        Parameters:
        - bboxes_A, labels_A, scores_A: Ground truth (original model output).
        - bboxes_B, labels_B, scores_B: Predictions (optimized model output).

        Returns:
        - mean Average Precision (mAP)
        """
        ap_per_class = []

        unique_classes = set(labels_A) | set(labels_B)  # Unique class labels in both models
        for cls in unique_classes:
            gt_boxes = [b for b, lbl in zip(bboxes_A, labels_A) if lbl == cls]
            pred_boxes = [(s, b) for b, lbl, s in zip(bboxes_B, labels_B, scores_B) if lbl == cls]

            # Sort predictions by confidence score (descending)
            pred_boxes.sort(key=lambda x: x[0], reverse=True)

            tp, fp = [], []
            gt_matched = set()

            for score, pred_box in pred_boxes:
                matched = False
                for i, gt_box in enumerate(gt_boxes):
                    if i in gt_matched:
                        continue  # Skip already matched GT boxes
                    iou = self.compute_iou(pred_box, gt_box)

                    if iou >= self.iou_threshold:
                        tp.append(1)
                        fp.append(0)
                        gt_matched.add(i)
                        matched = True
                        break
                if not matched:
                    tp.append(0)
                    fp.append(1)

            tp = np.cumsum(tp)
            fp = np.cumsum(fp)
            recall = tp / max(len(gt_boxes), 1)
            precision = tp / (tp + fp + 1e-9)  # Avoid division by zero
            #print ("P:" + str(precision))
            #print ("R" + str(recall))
            # ap = np.trapz(precision, recall)  # Compute area under PR curve
        return (2 * (precision * recall)) / (precision + recall)    
            
            #ap_per_class.append(ap)

        #print(ap_per_class)
        #return np.mean(ap_per_class) if ap_per_class else 0.0


import numpy as np

class SSDObjectDetectionEvaluator:
    def __init__(self, iou_threshold=0.5):
        """
        Initialize the evaluator with IoU threshold.
        """
        self.iou_threshold = iou_threshold

    def compute_iou(self, box1, box2):
        """
        Compute IoU (Intersection over Union) between two bounding boxes.
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        inter_x1 = max(x1, x1g)
        inter_y1 = max(y1, y1g)
        inter_x2 = min(x2, x2g)
        inter_y2 = min(y2, y2g)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def compute_f1(self, bboxes_A, labels_A, scores_A, bboxes_B, labels_B, scores_B):
        """
        Compute F1 score by comparing model B (optimized) predictions against model A (original) as pseudo-ground truth.
        
        Parameters:
        - bboxes_A, labels_A, scores_A: Ground truth (original model output).
        - bboxes_B, labels_B, scores_B: Predictions (optimized model output).

        Returns:
        - F1 score (mean F1 across all classes)
        """
        f1_per_class = []

        unique_classes = set(labels_A) | set(labels_B)  # Unique class labels in both models
        for cls in unique_classes:
            gt_boxes = [b for b, lbl in zip(bboxes_A, labels_A) if lbl == cls]
            pred_boxes = [(s, b) for b, lbl, s in zip(bboxes_B, labels_B, scores_B) if lbl == cls]

            # Sort predictions by confidence score (descending)
            pred_boxes.sort(key=lambda x: x[0], reverse=True)

            tp, fp, fn = 0, 0, 0
            gt_matched = set()

            for score, pred_box in pred_boxes:
                matched = False
                for i, gt_box in enumerate(gt_boxes):
                    if i in gt_matched:
                        continue  # Skip already matched GT boxes
                    iou = self.compute_iou(pred_box, gt_box)
                    if iou >= self.iou_threshold:
                        tp += 1
                        gt_matched.add(i)
                        matched = True
                        break
                if not matched:
                    fp += 1  # False positive

            fn = len(gt_boxes) - tp  # False negatives

            # Calculate precision and recall for the class
            precision = tp / (tp + fp + 1e-9)  # Avoid division by zero
            recall = tp / (tp + fn + 1e-9)    # Avoid division by zero

            # Calculate F1 score for the class
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0  # If both precision and recall are 0, F1 is 0

            f1_per_class.append(f1)

        return np.mean(f1_per_class) if f1_per_class else 0.0  # Return mean F1 across all classes

