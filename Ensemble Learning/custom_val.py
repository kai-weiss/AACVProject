import copy

import torch
from ultralytics import YOLO

import ensemble_learning
from Hierarchical_classification.custom_validator import CustomDetectionValidator


def sort_results_by_confidence(results):
    sorted_results_list = []

    for result in results:
        # Deep copy to avoid modifying the original result
        sorted_result = copy.deepcopy(result)

        # Extract the boxes object
        boxes = sorted_result.boxes

        if boxes is not None and len(boxes) > 0:
            # Extract the confidence scores
            conf = boxes.conf

            # Get the sorted indices in descending order of confidence
            sorted_indices = torch.argsort(conf, descending=True)

            # Apply the sorted indices to all relevant box attributes
            boxes.data = boxes.data[sorted_indices]

            # Update the boxes in the sorted_result
            sorted_result.boxes = boxes

        # Append the sorted_result to the list
        sorted_results_list.append(sorted_result)

    return sorted_results_list


if __name__ == '__main__':
    # Initialize Model1 and Model2 with their weights
    model1 = YOLO(r"D:\AACV\Ensemble Learning\ensemble models\model1.pt")
    model2 = YOLO(r"D:\AACV\Ensemble Learning\ensemble models\model2.pt")

    print("model1", model1.names)
    print("model2", model2.names)

    # Individual predictions from Model1 and Model2
    print("\nModel1 predictions:")
    model1_predictions = model1.predict(r"D:\AACV\Ensemble Learning\data\val\images")
    print("\nModel2 predictions:")
    model2_predictions = model2.predict(r"D:\AACV\Ensemble Learning\data\val\images")

    # Remapping
    class_id_mapping = {0: 7, 1: 8, 2: 9}
    # IMPORTANT! DO NOT MISS THIS STEP  -- Remap the classIDs in model2_predictions
    remapped_model2_predictions = ensemble_learning.remap_predictions(model2_predictions, class_id_mapping)

    print("\nEnsemble Learning: Predictions:")
    # IMPORTANT!
    # Always send model1_predictions (bigger model) as the first arg
    # Always send remapped_model2_predictions (bigger model) as the second arg
    predictions = ensemble_learning.ensemble_predictions(model1_predictions, remapped_model2_predictions)

    print('Sorting ensemble predictions as per conf in desc order')
    sorted_predictions = sort_results_by_confidence(predictions)

    # Validation
    print("\nEnsemble Learning: Validation:")
    args = dict(model=r"D:\AACV\Ensemble Learning\ensemble models\model1.pt",
                data=r'D:\AACV\Ensemble Learning\data\idd_data.yaml')

    validator = CustomDetectionValidator(args=args, custom_predictions=sorted_predictions)
    results = validator()
