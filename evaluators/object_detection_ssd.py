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
        self.compute_metrics(bboxes_A, labels_A, scores_A, bboxes_B, labels_B, scores_B)["F1"]

    def compute_ap(self, precisions, recalls):
        """
        Compute Average Precision (AP) using multiple recall points.
        """
        recall_levels = np.linspace(0, 1, 11)  # 11 recall levels
        interpolated_precisions = []

        for recall_level in recall_levels:
            # Find max precision at recall >= recall_level
            max_prec = max([prec for prec, rec in zip(precisions, recalls) if rec >= recall_level], default=0)
            interpolated_precisions.append(max_prec)

        return np.mean(interpolated_precisions)

    def calculate_mAP(self, bboxes_A, labels_A, bboxes_B, labels_B, scores_B):
        """
        Calculate mean Average Precision (mAP) for object detection.

        Args:
            bboxes_A (list): Ground truth bounding boxes for model A.
            labels_A (list): Ground truth labels for model A.
            bboxes_B (list): Predicted bounding boxes for model B.
            labels_B (list): Predicted labels for model B.
            scores_B (list): Confidence scores for model B's predictions.

        Returns:
            float: mean Average Precision (mAP).
        """
        
        unique_classes = set(labels_A) | set(labels_B)  # Unique class labels in both models
        average_precisions = []  # List to store average precision per class
        
        # Iterate through each class
        for cls in unique_classes:
            gt_boxes = [b for b, lbl in zip(bboxes_A, labels_A) if lbl == cls]
            pred_boxes = [(s, b) for b, lbl, s in zip(bboxes_B, labels_B, scores_B) if lbl == cls]

            # Sort predictions by confidence score (descending)
            pred_boxes.sort(key=lambda x: x[0], reverse=True)

            tp, fp, fn = 0, 0, 0
            gt_matched = set()

            # Variables to store precision and recall at various thresholds
            precisions = []
            recalls = []
            thresholds = np.linspace(0, 1, num=101)  # 101 thresholds from 0 to 1

            for threshold in thresholds:
                tp, fp, fn = 0, 0, 0
                gt_matched = set()

                for score, pred_box in pred_boxes:
                    matched = False
                    for i, gt_box in enumerate(gt_boxes):
                        if i in gt_matched:
                            continue  # Skip already matched GT boxes
                        iou = self.compute_iou(pred_box, gt_box)
                        if iou >= self.iou_threshold:  # Thresholding based on IoU
                            tp += 1
                            gt_matched.add(i)
                            matched = True
                            break
                    if not matched:
                        fp += 1  # False positive

                fn = len(gt_boxes) - tp  # False negatives

                # Calculate precision and recall at this threshold
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)

                precisions.append(precision)
                recalls.append(recall)

            sorted_indices = np.argsort(recalls)
            recalls = np.array(recalls)[sorted_indices]
            recalls = np.maximum.accumulate(recalls)
            np.append(recalls, 1.0)
            np.append(precisions, min(precisions))  # Use lowest precision seen so far
            precisions = np.array(precisions)[sorted_indices]
            ap = self.compute_ap(precisions, recalls)
            average_precisions.append(ap)

        # Calculate mean Average Precision (mAP)
        mAP = np.mean(average_precisions) if average_precisions else 0.0
        # print("mAP", mAP)
        return mAP


    def compute_metrics(self, bboxes_A, labels_A, scores_A, bboxes_B, labels_B, scores_B):
        """
        Compute F1 score by comparing model B (optimized) predictions against model A (original) as pseudo-ground truth.
        
        Parameters:
        - bboxes_A, labels_A, scores_A: Ground truth (original model output).
        - bboxes_B, labels_B, scores_B: Predictions (optimized model output).

        Returns:
        - F1 score (mean F1 across all classes)
        """
        f1_per_class = []
        prec_per_class = []
        recall_per_class = []

        unique_classes = set(labels_A) | set(labels_B)  # Unique class labels in both models
        total_iou = []
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
                    total_iou.append(iou)
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
            prec_per_class.append(precision)
            recall_per_class.append(recall)

        f1 = np.mean(f1_per_class) if f1_per_class else 0.0 
        IoU = np.mean(total_iou)

        mAP = self.calculate_mAP(bboxes_A, labels_A, bboxes_B, labels_B, scores_B)

        # print("MAP: ", mAP)

        return {
            "F1": f1,
            "precision": np.mean(prec_per_class),
            "recall": np.mean(recall_per_class),
            "mAP": mAP,
            "IoU": IoU
        }
