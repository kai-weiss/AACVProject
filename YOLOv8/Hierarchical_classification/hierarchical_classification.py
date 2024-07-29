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

species_color_mapping = {
    0: "#FCF7FF",
    1: "#FF924C",
    2: "#E85155",
    3: "#FFCA3A",
    4: "#4267AC",
    5: '#8AC926',
    6: '#A64692',
    7: "#6A4C93",
    8: "#A1E8CC",
    9: "#4B4237",
    10: '#C5CA30',
    11: "#41D3BD",
    12: '#E067A7',
    13: "#FF595E",
    14: "#BFD7EA",
    15: "#e197f0",
}


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


def plot_n_images_from_imageids_list(image_id_list, new_predictions, species_mapping, img_path, label_path, n=None):
    if not n: n = len(image_id_list)
    gt_data = load_ground_truths(label_path)
    # Iterate through predictions
    for image_id in image_id_list[:n]:  # Change the number of predictions you want to process

        # Find corresponding row in CSV

        print(image_id)
        print(gt_data[image_id])

        # gt_bboxes = eval(csv_row["bounding_boxes"].values[0])

        filtered_predictions = [prediction for prediction in new_predictions if prediction["image_id"] == image_id]

        # Initialize lists to store the extracted fields
        category_ids = []
        original_category_ids = []
        scores = []
        original_scores = []
        bboxes = []

        # Load image
        img = Image.open(img_path + '/' + image_id)
        img_width, img_height = img.size

        fig, ax = plt.subplots()
        ax.imshow(img)

        # Loop through filtered predictions to extract fields
        for prediction in filtered_predictions:
            category_id = prediction["category_id"]
            original_category_id = ""
            original_score = ""
            color = species_color_mapping[category_id]
            if "original_category_id" in prediction:
                original_category_id = prediction["original_category_id"]
                original_score = prediction["original_score"]
                color = species_color_mapping[category_id]

            score = prediction["score"]
            bbox = prediction["bbox"]

            category_ids.append(category_id)
            scores.append(score)
            bboxes.append(bbox)

            original_category_ids.append(original_category_id)
            original_scores.append(original_score)

        if gt_data[image_id]:

            # Ground Truth boxes
            for bbox in gt_data[image_id]:
                print(bbox["bbox"])
                category_id = bbox['class']
                category_name = species_mapping.get(category_id, "Unknown")

                bbox = convert_box(bbox['bbox'], img_width, img_height)

                rect = patches.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2],
                    bbox[3],
                    linewidth=5,
                    edgecolor='g',
                    alpha=0.5,
                    label=f"GT: {category_name}",
                    facecolor='none')

                ax.add_patch(rect)

                # Show class on plot
                plt.text(
                    bbox[0],  # + bbox[2] - 100,#-220,
                    bbox[1] - 130,  # +bbox[3]+100,
                    f"{category_name}",
                    color="k",
                    backgroundcolor="g",
                    fontsize=10,
                    alpha=0.8,
                )

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
                edgecolor=color,
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
                fontsize=10,
                backgroundcolor=color,
            )

        # title = f"Path: {image_id}"  # \nClass: {category_name}"
        # ax.set_title(title)

        ax.legend()

        # Show plot
        plt.axis("off")
        plt.show()
        # savepath = r"C:\Users\kaiwe\Documents\Master\Semester 3\results/" + image_id
        # plt.savefig(savepath, bbox_inches='tight')


# Convert bbox from [xyxy] to [x1, y1, width, height]
def convert_box(bbox, img_width, img_height):
    x_center, y_center, width, height = bbox
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    width = width * img_width
    height = height * img_height

    return x_min, y_min, width, height


def unique_imagesids_from_predictions(predictions_data):
    unique_image_ids = []
    [unique_image_ids.append(prediction["image_id"]) for prediction in predictions_data if
     prediction["image_id"] not in unique_image_ids]

    return unique_image_ids


def load_predictions(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Filter out instances with "original_score" key
    # filtered_data = [item for item in data if "original_score" not in item]

    return data


# Load YOLO ground truth files
def load_ground_truths(yolo_dir):
    labels_dict = {}

    # Iterate over all files in the directory
    for filename in os.listdir(yolo_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(yolo_dir, filename)

            with open(file_path, 'r') as file:
                # Read all lines from the file
                lines = file.readlines()
                labels = []
                for line in lines:
                    if line.strip():
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:]]
                        labels.append({"class": class_id, "bbox": bbox})

                # Save the labels in the dictionary with the filename as the key
                labels_dict[os.path.splitext(filename)[0] + '.jpg'] = labels

    return labels_dict
