import numpy as np

class YOLOV3Extractor:
    def __init__(self):
        pass
    
    def extract_yolo_outputs(self, model_output, num_classes=80, conf_threshold=0.5):
        """
        Extracts the bounding boxes, labels, and scores from YOLOv3 model output (works for Tiny-YOLOv3).
        
        Args:
            model_output (list of tensors): List containing the output tensors from YOLOv3-based model. 
                                             It contains the bounding boxes, class scores, and objectness.
            num_classes (int): The number of classes the model is trained on.
            conf_threshold (float): Confidence threshold to filter predictions.

        Returns:
            tuple: A tuple containing:
                - bboxes (list): Bounding box coordinates, shape [num_boxes, 4].
                - labels (list): Predicted class labels for the boxes, shape [num_boxes].
                - scores (list): Confidence scores for each prediction, shape [num_boxes].
        """
        
        # Unpack the model outputs
        bboxes = model_output[0]  # Tensor: [1, ?, 4] - Bounding boxes (x, y, width, height)
        class_scores = model_output[1]  # Tensor: [1, 80, ?] - Class scores (80 classes)
        objectness = model_output[2]  # Tensor: [1, ?, 3] - Objectness scores (3 anchors per grid cell)

        # Apply the objectness score to filter the boxes with low confidence
        objectness_scores = objectness[..., 0]  # Get the objectness score for each box (1st index)

        # Apply confidence threshold to objectness scores
        mask = objectness_scores > conf_threshold  # Mask out boxes with low objectness

        # Get filtered bounding boxes and class scores (only those with confidence above threshold)
        if np.all(mask) != False:
            bboxes = bboxes[mask]
            objectness_scores = objectness_scores[mask]
            class_scores = class_scores[mask]
        else:
            return [], [], []
        # Use np.max() to get the max class score and corresponding label
        scores = np.max(class_scores, axis=1)
        labels = np.argmax(class_scores, axis=1)

        # Convert bboxes, labels, and scores to simple lists
        bboxes = bboxes.tolist()
        labels = labels.squeeze().tolist()
        scores = scores.tolist()

        # Return the filtered bounding boxes, labels, and scores as simple lists
        return bboxes, labels, scores  # Returning as simple lists
