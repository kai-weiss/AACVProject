import math
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import json
from collections import defaultdict
import numpy as np
import torch
from torch import inf, SymInt, Tensor
from ultralytics.utils.metrics import compute_ap
import math
import warnings
from pathlib import Path
import torch

from ultralytics.utils import LOGGER, SimpleClass, TryExcept, plt_settings


# Create the inverted dictionaries
def prepare_dictionaries(species_mapping, family_mapping):
    inverted_species_mapping = {value: key for key, value in species_mapping.items()}

    inverted_family_mapping = {}
    for key, value in family_mapping.items():
        if value:
            if value not in inverted_family_mapping:
                inverted_family_mapping[value] = [key]
            else:
                inverted_family_mapping[value].append(key)

    return inverted_species_mapping, inverted_family_mapping


def run_hierarchical_classification(predictions_file, threshold, species_mapping, family_mapping):
    # Load the predictions
    with open(predictions_file, "r") as f:
        predictions = json.load(f)

    inverted_species_mapping, inverted_family_mapping = prepare_dictionaries(species_mapping, family_mapping)

    # Process each prediction and update the category_id and original_category_id
    for prediction in predictions:

        class_found = False

        species_index = prediction['category_id']
        species_name = species_mapping.get(species_index)

        original_score = max(prediction['activations'])

        if species_name is not None:
            # Determine if the species should be updated based on activation threshold
            if original_score < threshold:
                family_name = family_mapping.get(species_name)

                if family_name:
                    family_species_names = inverted_family_mapping[family_name]
                    # Calculate upper category score
                    family_score = 0
                    for family_specie_name in family_species_names:
                        family_specie_id = inverted_species_mapping[family_specie_name]
                        family_score += prediction['activations'][family_specie_id]

                    family_index = inverted_species_mapping[family_name]

                    if family_score >= threshold:
                        class_found = True
                        # Update category_id and original_category_id
                        prediction['original_category_id'] = species_index
                        prediction['category_id'] = family_index
                        prediction['original_score'] = original_score
                        prediction['score'] = family_score

                if not class_found:
                    # if there is an upper category
                    prediction['original_category_id'] = species_index
                    prediction['category_id'] = 15  # root
                    prediction['original_score'] = original_score
                    prediction['score'] = 1

    return predictions


def plot_with_new_predictions(input_path, new_predictions, species_mapping):
    img_paths = []

    # Traverse directory
    for root, dirs, files in os.walk(input_path):
        for file in files:
            full_path = os.path.join(root, file)
            img_paths.append(full_path)

    # print(img_paths)

    for img_path in img_paths:

        # Initialize lists to store the extracted fields
        category_ids = []
        original_category_ids = []
        scores = []
        original_scores = []
        bboxes = []

        for prediction in new_predictions:
            category_id = prediction["category_id"]
            original_category_id = ""
            original_score = ""

            if "original_category_id" in prediction:
                original_category_id = prediction["original_category_id"]
                original_score = prediction["original_score"]

            score = prediction["score"]
            bbox = prediction["bbox"]

            category_ids.append(category_id)
            scores.append(score)
            bboxes.append(bbox)

            original_category_ids.append(original_category_id)
            original_scores.append(original_score)

        # Load image
        img = Image.open(img_path)

        fig, ax = plt.subplots()
        ax.imshow(img)

        # Predicted boxes
        for idx, bbox in enumerate(bboxes):

            category_id = category_ids[idx]
            original_category_id = original_category_ids[idx]
            original_score = original_scores[idx]
            original_category_name = species_mapping.get(original_category_id, "Unknown")
            category_name = species_mapping.get(category_id, "Unknown")
            score = scores[idx]

            if original_category_id != "":
                label = f"{category_name}, {score:.2f} ({original_category_name}, {original_score:.2f})"
            else:
                label = f"{category_name} ({score:.2f})"

            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                linewidth=2,
                alpha=0.8,
                facecolor='none',
                label=label
            )

            ax.add_patch(rect)

            # Show class and score on plot
            plt.text(
                bbox[0],
                bbox[1] - 35,
                label,
                alpha=0.8,
                color="k",
                fontsize=7,
            )

        # Set title
        title = f"Path: {img_path}"  # \nClass: {category_name}"
        ax.set_title(title)

        ax.legend()

        # Show plot
        plt.axis("off")
        plt.show()


def load_predictions(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Filter out instances with "original_score" key
    # filtered_data = [item for item in data if "original_score" not in item]

    return data


# Load YOLO ground truth files
def load_ground_truths(yolo_dir):
    ground_truths = {}
    for file in os.listdir(yolo_dir):
        if file.endswith(".txt"):
            image_id = file.replace(".txt", ".jpg")
            with open(os.path.join(yolo_dir, file), 'r') as f:
                lines = f.readlines()
                boxes = [list(map(float, line.strip().split())) for line in lines]
                ground_truths[image_id] = boxes
    return ground_truths


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=True, CIoU=True, eps=1e-7):
    """
    Calculate IoU between two bounding boxes.
    Bounding boxes are expected to be in (x, y, width, height) format.
    """

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersect_x1 = max(x1, x2)
    intersect_y1 = max(y1, y2)
    intersect_x2 = min(x1 + w1, x2 + w2)
    intersect_y2 = min(y1 + h1, y2 + h2)

    intersect_area = max((intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1), 0)
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = intersect_area / float(box1_area + box2_area - intersect_area)

    return iou


def calculate_metrics(predictions, ground_truths, iou_threshold=0.5):
    class_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})

    for pred in predictions:

        image_id = pred["image_id"]
        pred_bbox = pred["bbox"]
        pred_category = pred["category_id"]
        gt_bboxes = ground_truths.get(image_id, [])

        matched = False
        for gt_bbox in gt_bboxes:
            gt_category, gt_xc, gt_yc, gt_w, gt_h = gt_bbox
            gt_bbox = [gt_xc, gt_yc, gt_w, gt_h]

            iou = bbox_iou(pred_bbox, gt_bbox)
            if iou >= iou_threshold:
                class_stats[pred_category]['TP'] += 1
                matched = True
                break

        if not matched:
            class_stats[pred_category]['FP'] += 1

    # Calculate FN for each class
    for image_id, gt_bboxes in ground_truths.items():
        for gt_bbox in gt_bboxes:
            gt_category = int(gt_bbox[0])
            class_stats[gt_category]['FN'] += 1

    for i in range(0, 16):
        class_stats[i]['FN'] = class_stats[i]['FN'] - class_stats[i]['TP']


    metrics = {}
    for category, stats in class_stats.items():
        TP, FP, FN = stats['TP'], stats['FP'], stats['FN']
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        metrics[category] = {'precision': precision, 'recall': recall}

    # Calculate overall precision and recall
    total_TP = sum(stats['TP'] for stats in class_stats.values())
    total_FP = sum(stats['FP'] for stats in class_stats.values())
    total_FN = sum(stats['FN'] for stats in class_stats.values())
    print(total_TP, total_FP, total_FN)
    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0

    print('##')
    for stats in class_stats:
        print(stats, class_stats[stats].values())
    print('##')
    return metrics, overall_precision, overall_recall


def calculate_map(predictions, ground_truths):
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    map_per_class = defaultdict(list)

    for iou_threshold in iou_thresholds:
        metrics, _, _ = calculate_metrics(predictions, ground_truths, iou_threshold)
        for category, metric in metrics.items():
            map_per_class[category].append(metric['precision'])

    map50 = {category: precisions[0] for category, precisions in map_per_class.items()}  # mAP@50
    map50_95 = {category: np.mean(precisions) for category, precisions in map_per_class.items()}  # mAP@50:95

    # Calculate overall mAP50 and mAP50-95
    overall_map50 = np.mean(list(map50.values()))
    overall_map50_95 = np.mean(list(map50_95.values()))

    return map50, map50_95, overall_map50, overall_map50_95


# weights for [P, R, mAP@0.5, mAP@0.5:0.95] extracted from the YOLOv8 framework
def calculate_fitness(precision, recall, map_50, map_50_95, weights=[0.0, 0.0, 0.1, 0.9]):
    return weights[0] * precision + weights[1] * recall + weights[2] * map_50 + weights[3] * map_50_95


def predict_with_hierarchical_classification(prediction_path, yolo_label_dir):
    predictions = load_predictions(prediction_path)

    print(len(predictions))
    ground_truths = load_ground_truths(yolo_label_dir)
    print(len(ground_truths))
    # Calculate metrics
    metrics_per_class, overall_precision, overall_recall = calculate_metrics(predictions, ground_truths)
    map50, map50_95, overall_map50, overall_map50_95 = calculate_map(predictions, ground_truths)
    # fitness = calculate_fitness(precision, recall, map50, map50_95)

    map50 = sorted(map50.items())
    map50_95 = sorted(map50_95.items())

    counter = 0
    for image in ground_truths:
        counter += len(ground_truths.get(image, []))
    print("###")
    print(counter)

    for category, metrics in sorted(metrics_per_class.items()):
        print(f"Class {category} - Precision: {metrics['precision']}, Recall: {metrics['recall']}")

    print(f"mAP@50 per class: {map50}")
    print(f"mAP@50:95 per class: {map50_95}")

    # Print overall results
    print(f"Overall Precision: {overall_precision}")
    print(f"Overall Recall: {overall_recall}")
    print(f"Overall mAP@50: {overall_map50}")
    print(f"Overall mAP@50:95: {overall_map50_95}")


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def ap_per_class(
    tp, conf, pred_cls, target_cls, plot=False, on_plot=None, names=(), eps=1e-16, prefix=""
):
    """
    Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
        names (tuple, optional): Tuple of class names to plot PR curves. Defaults to an empty tuple.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

    Returns:
        (tuple): A tuple of six arrays and one array of unique classes, where:
            tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.Shape: (nc,).
            fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class. Shape: (nc,).
            p (np.ndarray): Precision values at threshold given by max F1 metric for each class. Shape: (nc,).
            r (np.ndarray): Recall values at threshold given by max F1 metric for each class. Shape: (nc,).
            f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class. Shape: (nc,).
            ap (np.ndarray): Average precision for each class at different IoU thresholds. Shape: (nc, 10).
            unique_classes (np.ndarray): An array of unique classes that have data. Shape: (nc,).
            p_curve (np.ndarray): Precision curves for each class. Shape: (nc, 1000).
            r_curve (np.ndarray): Recall curves for each class. Shape: (nc, 1000).
            f1_curve (np.ndarray): F1-score curves for each class. Shape: (nc, 1000).
            x (np.ndarray): X-axis values for the curves. Shape: (1000,).
            prec_values: Precision values at mAP@0.5 for each class. Shape: (nc, 1000).
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values)  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values















