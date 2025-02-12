import numpy as np

class YOLOV3ObjectDetectionEvaluator:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    def compute_iou(self, bbox1, bbox2):
        """
        Computes the Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1, bbox2 (list): Bounding boxes in the format [x_center, y_center, width, height].

        Returns:
            float: IoU value between 0 and 1.
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate the coordinates of the intersection box
        inter_x1 = max(x1 - w1 / 2, x2 - w2 / 2)
        inter_y1 = max(y1 - h1 / 2, y2 - h2 / 2)
        inter_x2 = min(x1 + w1 / 2, x2 + w2 / 2)
        inter_y2 = min(y1 + h1 / 2, y2 + h2 / 2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)

        # Area of intersection
        intersection_area = inter_w * inter_h

        # Area of both boxes
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2

        # Area of union
        union_area = bbox1_area + bbox2_area - intersection_area

        # IoU = intersection area / union area
        return intersection_area / union_area if union_area > 0 else 0

    def compute_f1(self, bboxes_A, labels_A, scores_A, bboxes_B, labels_B, scores_B):
        return self.compute_metrics(bboxes_A, labels_A, scores_A, bboxes_B, labels_B, scores_B)["F1"]

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

        # Iterate through all predictions in bboxes_A for each image in the batch
        for batch_idx in range(len(bboxes_A)):
            pred_bboxes = bboxes_A[batch_idx]  # Shape: [num_boxes, 4]
            pred_labels = labels_A#[batch_idx]  # Shape: [num_boxes]
            pred_scores = scores_A[batch_idx]  # Shape: [num_boxes]
            
            gt_bboxes = bboxes_B[batch_idx]  # Shape: [num_boxes, 4]
            gt_labels = labels_B#[batch_idx]  # Shape: [num_boxes]
            gt_scores = scores_B[batch_idx]  # Shape: [num_boxes]

            # Iterate through all predicted boxes
            for i, pred_bbox in enumerate(pred_bboxes):
                pred_label = pred_labels[i]
                pred_score = pred_scores[i]
                
                best_iou = 0
                best_gt_idx = -1
                
                # Find the ground truth box with the highest IoU
                for j, gt_bbox in enumerate(gt_bboxes):
                    if gt_labels[j] == pred_label:  # Match by class label
                        iou = self.compute_iou(pred_bbox, gt_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = j
                
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
                        iou = self.compute_iou(pred_bbox, gt_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_idx = i
                
                if best_iou <= self.iou_threshold:
                    # False Negative (ground truth box has no valid prediction)
                    false_negatives += 1

        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {
            "F1": f1_score,
            "precision": precision,
            "recall": recall
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
