import json
import warnings
from pathlib import Path

import numpy as np
import os
import pickle

import torch
import yaml
from ultralytics.utils.metrics import ap_per_class, DetMetrics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from YOLOv8.Hierarchical_classification.custom_validator import CustomDetectionValidator


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    cx = x1 + (x2 / 2.0)
    cy = y1 + (y2 / 2.0)
    w = x2
    h = y2

    return [cx, cy, w, h]


def preprocess_data(json_file, yolo_dir, iou_threshold=0.5):
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
            bbox_xywhn = xyxy_to_xywh(bbox)

            iou = calculate_iou(bbox_xywhn, [gt_x, gt_y, gt_w, gt_h])

            if iou >= iou_threshold and category_id == gt_category_id:
                # print(category_id)
                tp_list.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # True Positive
                target_cls_list.append(gt_category_id)
            else:
                tp_list.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # False Positive

            conf_list.append(score)
            pred_cls_list.append(category_id)
            # target_cls_list.append(gt_category_id)

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


def filter_by_image_id(data, image_id):
    return [instance for instance in data if instance["image_id"] == image_id]


# Example usage
json_file = 'predictions.json'
yolo_dir = 'C:/Users/kaiwe/Documents/Master/Semester 3/archive/shrunk/val/labels/'


with open(json_file, 'r') as f:
    extracted_predictions = json.load(f)

with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    names = config['names']

# print("####" + str(tp.shape[0]) + " " + str(tp.shape[1]))
# print("####" + str(conf.shape))
# print("####" + str(pred_cls.shape))
# print("####" + str(target_cls.shape))

# model1 = YOLO(r"best.pt")
# model1_predictions = model1.predict(r"C:\Users\kaiwe\Documents\Master\Semester 3\archive\shrunk\val\images")

with open('model1_predictions.pkl', 'rb') as f:
    model_predictions = pickle.load(f)

counter = 0
mistakes = 0

for pred in model_predictions:
    counter += len(pred.boxes.cls)
    pred_filename = os.path.basename(pred.path)
    filtered_predictions = filter_by_image_id(extracted_predictions, pred_filename)

    # Potential errors, that would occur if we would miss some extracted predictions
    # Keep in mind, that ideally the variable 'mistakes' should always stay at 0
    if len(pred.boxes.cls) != len(filtered_predictions):
        mistakes += abs(len(pred.boxes.cls) - len(filtered_predictions))
        # print(len(pred.boxes.cls), len(filtered_predictions))

    # Here we are abusing the fact that the normal model predictions and the extracted predictions
    # are both sorted in the correct way already, so no need anymore to search for the exact preds
    for i in range(len(pred.boxes.cls)):
        # print(i, filtered_predictions[i]['score'])

        # Filter out errors
        if len(pred.boxes.cls) == len(filtered_predictions):

            # According error message added
            if filtered_predictions[i]['category_id'] != pred.boxes.cls[i]:
                warnings.warn('ERROR: If this prints, then the extraction failed')

            pred.boxes.conf[i] = filtered_predictions[i]['score']

print(counter)
print(mistakes)

args = dict(model=r"best.pt", data=r'C:\Users\kaiwe\Documents\Master\Semester 3\AACV\YOLOv8\config.yaml')

validator = CustomDetectionValidator(args=args, custom_predictions=None)
results = validator()


