import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import json


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
