import numpy as np

def iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0

def average_precision(recall, precision):
    """Compute Average Precision (AP) using interpolated precision-recall curve."""
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

def compute_map(model_A_preds, model_A_prime_preds, iou_threshold=0.5):
    """Compute mAP using Model A as pseudo-ground truth for Model A'."""
    aps = []
    for class_id in model_A_preds.keys():
        gt_bboxes = model_A_preds[class_id]  # Model A (pseudo-GT)
        pred_bboxes = sorted(model_A_prime_preds.get(class_id, []), key=lambda x: x[0], reverse=True)

        tp = np.zeros(len(pred_bboxes))
        fp = np.zeros(len(pred_bboxes))
        detected = []

        for i, (conf, pred_box) in enumerate(pred_bboxes):
            ious = [iou(pred_box, gt_box) for gt_box in gt_bboxes]
            max_iou = max(ious) if ious else 0
            max_index = np.argmax(ious) if ious else -1

            if max_iou >= iou_threshold and max_index not in detected:
                tp[i] = 1
                detected.append(max_index)
            else:
                fp[i] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / len(gt_bboxes) if gt_bboxes else np.array([])
        precision = cum_tp / (cum_tp + cum_fp)

        ap = average_precision(recall, precision)
        aps.append(ap)

    return np.mean(aps) if aps else 0

# Example Usage
# model_A_preds = {
#     1: [[50, 50, 150, 150]],  # Pseudo-ground truth from Model A
#     2: [[200, 200, 300, 300]]
# }

# model_A_prime_preds = {
#     1: [(0.9, [55, 55, 145, 145]), (0.8, [30, 30, 120, 120])],  # Predictions from Model A'
#     2: [(0.7, [210, 210, 290, 290])]
# }

# mAP = compute_map(model_A_preds, model_A_prime_preds)
# print("mAP between A and A':", mAP)
