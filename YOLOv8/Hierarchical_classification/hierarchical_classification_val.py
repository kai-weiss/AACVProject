import json
import numpy as np
import os

import yaml
from ultralytics.utils.metrics import ap_per_class


def preprocess_data(json_file, yolo_dir):
    # Load JSON data
    with open(json_file, 'r') as f:
        detections = json.load(f)

    # Initialize lists to collect data
    tp_list, conf_list, pred_cls_list, target_cls_list = [], [], [], []

    # Process each detection
    for det in detections:
        image_id = det['image_id']
        category_id = det['category_id']
        bbox = det['bbox']
        score = det['score']

        # Load corresponding YOLO ground truth
        yolo_file = os.path.join(yolo_dir, f"{image_id.split('.')[0]}.txt")
        if not os.path.exists(yolo_file):
            continue
        with open(yolo_file, 'r') as f:
            gt_data = f.readlines()

        # Parse YOLO ground truth
        for line in gt_data:
            gt_category_id, gt_x, gt_y, gt_w, gt_h = map(float, line.split())
            gt_category_id = int(gt_category_id)

            # Calculate Intersection over Union (IoU)
            # iou = calculate_iou(bbox, [gt_x, gt_y, gt_w, gt_h])

            if category_id == gt_category_id:
                print(category_id)
                tp_list.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # True Positive
            else:
                tp_list.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # False Positive

            conf_list.append(score)
            pred_cls_list.append(category_id)
            target_cls_list.append(gt_category_id)

    # Convert lists to numpy arrays
    tp = np.array(tp_list, dtype=bool)
    conf = np.array(conf_list)
    pred_cls = np.array(pred_cls_list)
    target_cls = np.array(target_cls_list)

    return tp, conf, pred_cls, target_cls


def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    intersect_x1 = max(x1, x2)
    intersect_y1 = max(y1, y2)
    intersect_x2 = min(x1 + w1, x2 + w2)
    intersect_y2 = min(y1 + h1, y2 + h2)

    intersect_area = max((intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1), 0)
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = intersect_area / float(box1_area + box2_area - intersect_area)

    return iou


# Example usage
json_file = 'predictions.json'
yolo_dir = 'C:/Users/kaiwe/Documents/Master/Semester 3/archive/shrunk/val/labels/'

tp, conf, pred_cls, target_cls = preprocess_data(json_file, yolo_dir)

with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    names = config['names']

results = ap_per_class(tp, conf, pred_cls, target_cls, names=names)

# Process results as needed
print((results[0]))
print((results[2]))
