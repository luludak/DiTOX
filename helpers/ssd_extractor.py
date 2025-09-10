import numpy as np

class SSDExtractor:

    def __init__(self):
        pass

    def extract_ssd_outputs(self, output, conf_threshold=0.1):
        bboxes, labels, scores = output
        bboxes = np.squeeze(bboxes) # (nbox, 4)
        labels = np.squeeze(labels) # (nbox,)
        scores = np.squeeze(scores) # (nbox,)

        # Apply confidence threshold
        mask = scores >= conf_threshold
        bboxes = bboxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        bounding_boxes = [tuple(box) for box in bboxes]
        class_ids = labels.astype(int).tolist()
        confidences = scores.astype(float).tolist()

        return bounding_boxes, class_ids, confidences