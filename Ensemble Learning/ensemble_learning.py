import copy

import numpy as np
import torch
from ultralytics.engine.results import Results


def remap_class_ids_in_boxes(boxes, mapping):
    # Clone the boxes to avoid modifying the original
    new_boxes = copy.deepcopy(boxes)

    # Remap the class IDs within the tensor directly
    remapped_classes = torch.tensor([mapping[int(cls.item())] for cls in new_boxes.cls], device=new_boxes.cls.device)

    # Update the data tensor to reflect the remapped class IDs
    new_boxes.data[:, -1] = remapped_classes

    return new_boxes


def remap_predictions(predictions, mapping):
    remapped_predictions = []

    for result in predictions:
        # Clone the result to avoid modifying the original
        new_result = copy.deepcopy(result)

        # Remap the class IDs in the boxes
        new_result.boxes = remap_class_ids_in_boxes(new_result.boxes, mapping)

        remapped_predictions.append(new_result)

    return remapped_predictions


def iou(box1, box2):
    """
    Custom NMS (post-processing function) -- inspired from YOLOv8 implementation
    """
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)

    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)

    return inter_area / float(box1_area + box2_area - inter_area)


def ensemble_predictions(results1, results2, iou_threshold=0.6):
    """
    Function to merge the predictions from both models
    :param results1: always the predictions from the bigger model
    :param results2: always the (remapped) predictions from the smaller model
    """

    final_results_list = []

    assert len(results1) == len(results2), "The number of images in results1 and results2 must be the same."

    for img_idx in range(len(results1)):
        # print("img_idx: ", img_idx)
        img_final_boxes = []
        img_final_scores = []
        img_final_classes = []

        device = results1[img_idx].boxes.xyxy.device
        orig_shape = results1[img_idx].boxes.orig_shape

        # Extract predictions for the current image from both results1 (model1) and results2 (model2)
        # model 1
        boxes1 = results1[img_idx].boxes.xyxy.cpu().numpy()
        scores1 = results1[img_idx].boxes.conf.cpu().numpy()
        classes1 = results1[img_idx].boxes.cls.cpu().numpy()
        # model 2
        boxes2 = results2[img_idx].boxes.xyxy.cpu().numpy()
        scores2 = results2[img_idx].boxes.conf.cpu().numpy()
        classes2 = results2[img_idx].boxes.cls.cpu().numpy()

        # Use Model 1's predictions for class IDs 0-6
        for i in range(len(classes1)):
            if classes1[i] < 7:
                img_final_boxes.append(boxes1[i])
                img_final_scores.append(scores1[i])
                img_final_classes.append(classes1[i])

        # Combine predictions from both models for class IDs 7-9
        combined_boxes = []
        combined_scores = []
        combined_classes = []

        # model 1
        for i in range(len(classes1)):
            if classes1[i] >= 7:
                combined_boxes.append(boxes1[i])
                combined_scores.append(scores1[i])
                combined_classes.append(classes1[i])

        # model 2
        for i in range(len(classes2)):
            combined_boxes.append(boxes2[i])
            combined_scores.append(scores2[i])
            combined_classes.append(classes2[i])

        # Apply NMS to combined predictions for class IDs 7-9
        # TODO: check if we can use the NMS operation defined in metrics.py here
        combined_indices = list(range(len(combined_boxes)))
        for i in range(len(combined_boxes)):
            if combined_scores[i] == 0:
                continue
            for j in range(i + 1, len(combined_boxes)):
                if combined_scores[j] == 0:
                    continue
                if iou(combined_boxes[i][:4], combined_boxes[j][:4]) > iou_threshold:
                    if combined_scores[i] >= combined_scores[j]:
                        combined_scores[j] = 0
                    else:
                        combined_scores[i] = 0

        for i in combined_indices:
            if combined_scores[i] > 0:  # conf threshold is set to 0
                # TODO: check the conf threshold set by YOLO for both val and predict tasks
                img_final_boxes.append(combined_boxes[i])
                img_final_scores.append(combined_scores[i])
                img_final_classes.append(combined_classes[i])

        # Create Boxes.data
        if len(img_final_boxes) == 0:  # no detections
            # print(f"No detections for image {img_idx}. Creating empty Boxes object.")
            final_data = torch.empty((0, 6), device=device)
        else:  # detections present
            # Convert lists to numpy arrays
            img_final_boxes = np.array(img_final_boxes)
            img_final_scores = np.array(img_final_scores)
            img_final_classes = np.array(img_final_classes)
            # Convert to tensors and move to the original device
            img_final_boxes = torch.tensor(img_final_boxes, device=device)
            img_final_scores = torch.tensor(img_final_scores, device=device)
            img_final_classes = torch.tensor(img_final_classes, device=device)
            # Create a single tensor containing all the data
            final_data = torch.cat([img_final_boxes, img_final_scores.unsqueeze(1), img_final_classes.unsqueeze(1)],
                                   dim=1)

        # Create Results object
        final_results = Results(
            orig_img=results1[img_idx].orig_img,
            path=results1[img_idx].path,
            names=results1[img_idx].names,
            boxes=final_data
        )
        # Combine Results objects into a list
        final_results_list.append(final_results)

    return final_results_list
