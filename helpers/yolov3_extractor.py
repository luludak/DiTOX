import numpy as np

class YOLOV3Extractor:
    def __init__(self):
        pass

    def extract_yolo_outputs(self, model_output, conf_threshold=0.1):
        """
        Extract all YOLOv3 ONNX raw outputs into parallel lists:
        boxes, labels, scores, without relying on NMS indices.

        Args:
            model_output: Tuple of (boxes_raw, scores_raw, indices)
                - boxes_raw: [batch, num_boxes, 4]
                - scores_raw: [batch, num_classes, num_boxes]
                - indices: ignored
            conf_threshold: minimum confidence score to keep a detection

        Returns:
            boxes, labels, scores: lists
        """
        boxes_raw, scores_raw, _ = model_output

        boxes, labels, scores = [], [], []

        batch_size = boxes_raw.shape[0]
        num_classes = scores_raw.shape[1]
        num_boxes = boxes_raw.shape[1]

        for batch in range(batch_size):
            for box_id in range(num_boxes):
                for cls_id in range(num_classes):
                    score = float(scores_raw[batch, cls_id, box_id])
                    if score >= conf_threshold:
                        box = boxes_raw[batch, box_id].tolist()
                        boxes.append(box)
                        labels.append(cls_id)
                        scores.append(score)

        return boxes, labels, scores
    
#------------ Deprecated Class: Used to retrieve detected labels, not all boxes ------
# import numpy as np

# class YOLOV3Extractor:
#     def __init__(self):
#         pass

#     def extract_yolo_outputs(self, model_output, conf_threshold=0.1):
#         """
#         Extract YOLOv3 ONNX raw outputs into parallel lists:
#         boxes, labels, scores, using the indices array from NMS.
#         """
#         boxes_raw, scores_raw, indices = model_output  # indices: int32[1, M, 3] or int32[M,3]

#         boxes, labels, scores = [], [], []

#         # If batch dimension missing in indices, add it
#         if indices.ndim == 2:
#             indices = np.expand_dims(indices, axis=0)  # [1, M, 3]

#         # Loop over batch dimension (usually batch=1)
#         for batch in range(boxes_raw.shape[0]):
#             if indices.shape[1] == 0:
#                 continue  # no boxes selected

#             # Iterate over rows in indices[batch]
#             for idx_row in indices[batch]:
#                 batch_idx = int(idx_row[0])
#                 cls_id = int(idx_row[1])
#                 box_id = int(idx_row[2])

#                 box = boxes_raw[batch_idx, box_id].tolist()
#                 score = float(scores_raw[batch_idx, cls_id, box_id])
                
#                 if score >= conf_threshold:
#                     boxes.append(box)
#                     labels.append(cls_id)
#                     scores.append(score)

#         return boxes, labels, scores