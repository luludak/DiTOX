import numpy as np

class YOLOV2Extractor:

    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        pass

    def extract_yolo_outputs(self, output_tensor):
        """
        Extract bounding boxes, labels, and scores from YOLOv2 output tensor.

        Parameters:
        - output_tensor: NumPy array of shape (1, 425, 13, 13) from YOLOv2.

        Returns:
        - bboxes: List of bounding boxes in (x_min, y_min, x_max, y_max) format.
        - labels: List of class labels.
        - scores: List of confidence scores.
        """
        output_tensor = np.transpose(np.squeeze(output_tensor[0]), (1, 2, 0))  # Convert to (13, 13, 425)
        
        S = 13  # Grid size
        B = 5   # Number of anchor boxes per grid cell
        C = 80  # Number of classes

        # Reshape into (S, S, B, 5 + C)
        output_tensor = output_tensor.reshape((S, S, B, 5 + C))

        bboxes, labels, scores = [], [], []

        for row in range(S):
            for col in range(S):
                for b in range(B):
                    tx, ty, tw, th, obj_score = output_tensor[row, col, b, :5]
                    class_probs = output_tensor[row, col, b, 5:]  # Class probabilities
                    
                    # Convert class probabilities to class score
                    class_probs = np.exp(class_probs) / np.sum(np.exp(class_probs))  # Softmax
                    max_class_index = np.argmax(class_probs)
                    max_class_score = class_probs[max_class_index]
                    
                    # Compute final confidence score
                    confidence = obj_score * max_class_score
                    if confidence < self.conf_threshold:
                        continue  # Skip low confidence detections

                    # Convert (tx, ty, tw, th) to (x_min, y_min, x_max, y_max)
                    bx = (col + tx) / S  # Convert cell-relative x to image scale
                    by = (row + ty) / S
                    bw = np.exp(tw) / S
                    bh = np.exp(th) / S

                    x_min = max(0, bx - bw / 2)
                    y_min = max(0, by - bh / 2)
                    x_max = min(1, bx + bw / 2)
                    y_max = min(1, by + bh / 2)

                    bboxes.append([x_min, y_min, x_max, y_max])
                    labels.append(max_class_index)
                    scores.append(confidence)

        return bboxes, labels, scores
