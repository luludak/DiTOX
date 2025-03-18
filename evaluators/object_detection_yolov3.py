import numpy as np

class YOLOV3ObjectDetectionEvaluator:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold


    def calculate_mAP(self, bboxes_A, labels_A, scores_A, bboxes_B, labels_B, scores_B):
        """
        Calculate mean Average Precision (mAP) for object detection with batch support.

        Args:
            bboxes_A (list): Ground truth bounding boxes for model A (batch).
            labels_A (list): Ground truth labels for model A (batch).
            bboxes_B (list): Predicted bounding boxes for model B (batch).
            labels_B (list): Predicted labels for model B (batch).
            scores_B (list): Confidence scores for model B's predictions (batch).

        Returns:
            float: mean Average Precision (mAP).
        """
        
        unique_classes = set(lbl for lbl in labels_A) | set(lbl for lbl in labels_B)  # Unique class labels in both models
        average_precisions = []  # List to store average precision per class

        # print(len(unique_classes))
        
        # Iterate through each class
        # all_iou = []
        for cls in unique_classes:
            all_gt_boxes = []
            all_pred_boxes = []

            # Collect ground truth and predicted boxes for this class across the entire batch
            for batch_idx in range(len(bboxes_A)):  # Iterate through the batch
                gt_bboxes = bboxes_A[batch_idx]  # Shape: [num_boxes, 4]
                gt_labels = labels_A#[batch_idx]  # Shape: [num_boxes]
                gt_scores = scores_A[batch_idx]  # Shape: [num_boxes]

                pred_bboxes = bboxes_B[batch_idx]  # Shape: [num_boxes, 4]
                pred_labels = labels_B#[batch_idx]  # Shape: [num_boxes]
                pred_scores = scores_B[batch_idx]  # Shape: [num_boxes]


                if len(pred_bboxes) != 0:
                    all_gt_boxes.extend(b for s, b, l in zip(gt_scores, gt_bboxes, gt_labels) if l == cls)
                    all_pred_boxes.extend((s, b) for s, b, l in zip(pred_scores, pred_bboxes, pred_labels) if l == cls) 


            tp, fp, fn = 0, 0, 0
            gt_matched = set()

            # Variables to store precision and recall at various thresholds
            precisions = []
            recalls = []
            thresholds = np.linspace(0, 1, num=101)  # 101 thresholds from 0 to 1

            for threshold in thresholds:
                tp, fp, fn = 0, 0, 0
                gt_matched = set()

                for score, pred_box in all_pred_boxes:
                    # print(pred_box)
                    matched = False
                    for i, gt_box in enumerate(all_gt_boxes):
                        if i in gt_matched:
                            continue  # Skip already matched GT boxes
                        iou = self.compute_iou(gt_box, pred_box)
                        # all_iou.append(iou)
                        if iou >= self.iou_threshold:  # Thresholding based on IoU
                            tp += 1
                            gt_matched.add(i)
                            matched = True
                            break
                    if not matched:
                        fp += 1  # False positive

                fn = len(all_gt_boxes) - tp  # False negatives

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
        # print("mAP:", mAP)
        # print(all_iou)
        return mAP


    def compute_iou(self, gt_box, pred_box):
        """
        Compute the Intersection over Union (IoU) between two bounding boxes.
        Assumes that boxes are in [x_min, y_min, x_max, y_max] format.

        Parameters:
        - pred_box: Predicted bounding box [x_min, y_min, x_max, y_max]
        - gt_box: Ground truth bounding box [x_min, y_min, x_max, y_max]

        Returns:
        - IoU: Intersection over Union value.
        """
        x_min_pred, y_min_pred, x_max_pred, y_max_pred = pred_box
        x_min_gt, y_min_gt, x_max_gt, y_max_gt = gt_box

        # Compute the area of intersection
        x_min_inter = max(x_min_pred, x_min_gt)
        y_min_inter = max(y_min_pred, y_min_gt)
        x_max_inter = min(x_max_pred, x_max_gt)
        y_max_inter = min(y_max_pred, y_max_gt)

        # If there is no overlap, return 0 IoU
        if x_min_inter >= x_max_inter or y_min_inter >= y_max_inter:
            return 0.0

        intersection_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)

        # Compute the area of both bounding boxes
        area_pred = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
        area_gt = (x_max_gt - x_min_gt) * (y_max_gt - y_min_gt)

        # Compute the area of union
        union_area = area_pred + area_gt - intersection_area

        # Compute the IoU
        iou = intersection_area / union_area
        return iou

    def compute_metrics(self, bboxes_A, labels_A, scores_A, bboxes_B, labels_B, scores_B):
        """
        Computes the F1 score for YOLOv3 model predictions.
        
        Args:
            bboxes_A (numpy.ndarray): Predicted bounding boxes in format [batch_size, num_boxes, 4].
            labels_A (list): Predicted class labels for each bounding box.
            scores_A (list): Confidence scores for each predicted bounding box.
            bboxes_B (numpy.ndarray): Ground truth bounding boxes in format [batch_size, num_boxes, 4].
            labels_B (list): Ground truth class labels for each bounding box.
            scores_B (list): Ground truth confidence scores (if applicable, for evaluation purposes).

        Returns:
            float: The F1 score of the object detection model.
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_iou = []

        f1_per_class = []
        prec_per_class = []
        recall_per_class = []

        # Iterate through all predictions in bboxes_A for each image in the batch
        for batch_idx in range(len(bboxes_A)):
            gt_bboxes = bboxes_A[batch_idx]  # Shape: [num_boxes, 4]
            gt_labels = labels_A#[batch_idx]  # Shape: [num_boxes]
            gt_scores = scores_A[batch_idx]  # Shape: [num_boxes]

            pred_bboxes = bboxes_B[batch_idx]  # Shape: [num_boxes, 4]
            pred_labels = labels_B#[batch_idx]  # Shape: [num_boxes]
            pred_scores = scores_B[batch_idx]  # Shape: [num_boxes]

            # Iterate through all predicted boxes
            for i, pred_bbox in enumerate(pred_bboxes):
                pred_label = pred_labels[i]
                pred_score = pred_scores[i]
                
                best_iou = 0
                best_gt_idx = -1
                
                # Find the ground truth box with the highest IoU
                for j, gt_bbox in enumerate(gt_bboxes):
                    if gt_labels[j] == pred_label:  # Match by class label
                        iou = self.compute_iou(gt_bbox, pred_bbox)
                        # Add it to the list only once.
                        
                        
                        if iou > best_iou:
                            
                            best_iou = iou
                            best_gt_idx = j
                
                total_iou.append(best_iou)
                if best_iou > self.iou_threshold:
                    # True Positive (match found)
                    true_positives += 1
                else:
                    # False Positive (no match or IoU is low)
                    false_positives += 1

            # Iterate through all ground truth boxes to count false negatives
            for j, gt_bbox in enumerate(gt_bboxes):
                gt_label = gt_labels[j]
                
                best_iou = 0
                best_pred_idx = -1
                
                for i, pred_bbox in enumerate(pred_bboxes):
                    if pred_labels[i] == gt_label:  # Match by class label
                        iou = self.compute_iou(gt_bbox, pred_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_idx = i
                
                if best_iou <= self.iou_threshold:
                    # False Negative (ground truth box has no valid prediction)
                    false_negatives += 1

            # Calculate precision and recall
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            prec_per_class.append(precision)
            recall_per_class.append(recall)
        
            # Calculate F1 score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_per_class.append(f1_score)
        
        mIoU = np.mean([i for i in total_iou if i > 0])
        f1 = np.mean(f1_per_class) if f1_per_class else 0.0

        mAP = self.calculate_mAP(bboxes_A, labels_A, scores_A, bboxes_B, labels_B, scores_B)

        return {
            "F1": f1,
            "precision": np.mean(prec_per_class),
            "recall": np.mean(recall_per_class),
            "mAP": mAP,
            "IoU": mIoU
        }


# Example usage:
# evaluator = YOLOV3ObjectDetectionEvaluator(iou_threshold=0.5)

# Example bounding boxes, labels, and scores for predicted and ground truth data
# bboxes_A = np.array([[[0.5, 0.5, 0.2, 0.3], [0.6, 0.7, 0.2, 0.3]]])  # Predicted bounding boxes (batch_size=1, num_boxes=2, 4)
# labels_A = np.array([[0, 1]])  # Predicted class labels
# scores_A = np.array([[0.9, 0.8]])  # Predicted confidence scores

# bboxes_B = np.array([[[0.5, 0.5, 0.2, 0.3], [0.65, 0.75, 0.2, 0.3]]])  # Ground truth bounding boxes
# labels_B = np.array([[0, 1]])  # Ground truth class labels
# scores_B = np.array([[1.0, 1.0]])  # Ground truth confidence scores (if applicable)

# # Compute F1 score
# f1 = evaluator.compute_f1(bboxes_A, labels_A, scores_A, bboxes_B, labels_B, scores_B)
# print(f"F1 score: {f1:.4f}")
