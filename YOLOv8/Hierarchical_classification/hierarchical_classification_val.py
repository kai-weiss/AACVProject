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
from PIL import Image


def xywh_to_xywhn(box, img_width, img_height):
    x1, y1, w, h = box
    x_norm = x1 / img_width
    y_norm = y1 / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    return [x_norm, y_norm, w_norm, h_norm]


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
            # bbox_xywhn = xyxy_to_xywh(bbox)

            iou = calculate_iou(bbox, [gt_x, gt_y, gt_w, gt_h])

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


def get_family_name_from_family_id(family_id):
    return species_mapping[family_id]


# Since it is easier to just compare specific predictions in the validation function in YOLO's libary (instead of doing
# it the other way in the libary), we convert every family class into their specific class
def get_specific_prediction(prediction, bbox_xywhn, iou_threshold=0.5):
    # already specific class

    if prediction['category_id'] not in [10, 11, 12, 13, 14]:
        return prediction['category_id']

    image_id = prediction['image_id']

    # Load corresponding YOLO ground truth
    yolo_file = os.path.join(yolo_dir, f"{image_id.split('.')[0]}.txt")
    # yolo_img_path = os.path.join(yolo_dir_img, f"{image_id.split('.')[0]}.jpg")

    if not os.path.exists(yolo_file):
        return -1
    with open(yolo_file, 'r') as f:
        gt_data = f.readlines()

    # image = Image.open(yolo_img_path)
    # img_width, img_height = image.size
    best_iou = 0.0
    gt_category_id = None

    # Parse YOLO ground truth
    # IOU Calculation
    for line in gt_data:
        gt_category_id, gt_x, gt_y, gt_w, gt_h = map(float, line.split())

        # bbox_xywhn = xywh_to_xywhn(prediction["bbox"], img_width, img_height)



        iou = calculate_iou(bbox_xywhn, [gt_x, gt_y, gt_w, gt_h])

        if iou > best_iou:
            best_iou = iou

    if best_iou >= iou_threshold and gt_category_id is not None:
        return gt_category_id
    else:
        return prediction["original_category_id"]


# Example usage
json_file = 'selective_hierarchical_predictions.json'
yolo_dir = 'C:/Users/kaiwe/Documents/Master/Semester 3/archive/shrunk/val/labels/'
yolo_dir_img = 'C:/Users/kaiwe/Documents/Master/Semester 3/archive/shrunk/val/images/'

with open(json_file, 'r') as f:
    extracted_predictions = json.load(f)

with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    names = config['names']
    species_mapping = config['species_mapping']

# model = YOLO(r"best.pt")
# model1_predictions = model.predict(r"C:\Users\kaiwe\Documents\Master\Semester 3\archive\shrunk\val\images", conf=0.001, iou=0.6)

# with open('model_predictions.pickle', 'wb') as handle:
#    pickle.dump(model1_predictions, handle)

with open('model1_predictions.pkl', 'rb') as f:
    model_predictions = pickle.load(f)

counter = 0
mistakes = 0

for pred in model_predictions:
    counter += len(pred.boxes.cls)
    pred_filename = os.path.basename(pred.path)
    filtered_predictions = filter_by_image_id(extracted_predictions, pred_filename)

    # Potential errors, that would occur if we would miss some extracted predictions (can happen due to nms)
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
            if 'original_category_id' in filtered_predictions[i]:
                if filtered_predictions[i]['original_category_id'] != pred.boxes.cls[i]:
                    warnings.warn('ERROR: If this prints, then the extraction failed')
            elif filtered_predictions[i]['category_id'] != pred.boxes.cls[i]:
                warnings.warn('ERROR: If this prints, then the extraction failed')

            # Ignore root class
            if filtered_predictions[i]['category_id'] != 15:
                specific_class = get_specific_prediction(filtered_predictions[i], pred.boxes.xywhn[i])

                pred.boxes.cls[i] = specific_class
                pred.boxes.conf[i] = filtered_predictions[i]['score']

print(counter)
print(mistakes)

args = dict(model=r"best.pt", data=r'C:\Users\kaiwe\Documents\Master\Semester 3\AACV\YOLOv8\config.yaml', save_dir=True)

validator = CustomDetectionValidator(args=args, custom_predictions=model_predictions)
results = validator()
