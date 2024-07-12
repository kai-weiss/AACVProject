from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

import ensemble_learning

if __name__ == '__main__':
    # Initialize Model1 and Model2 with their weights
    model1 = YOLO(r"D:\AACV\Ensemble Learning\ensemble models\model1.pt")
    model2 = YOLO(r"D:\AACV\Ensemble Learning\ensemble models\model2.pt")

    # Individual predictions from Model1 and Model2
    print("\n Model1 predictions:")
    model1_predictions = model1.predict(r"D:\AACV\Ensemble Learning\data\val\images")
    print("\n Model2 predictions:")
    model2_predictions = model2.predict(r"D:\AACV\Ensemble Learning\data\val\images")

    # Remapping
    class_id_mapping = {0: 7, 1: 8, 2: 9}
    # IMPORTANT! DO NOT MISS THIS STEP  -- Remap the classIDs in model2_predictions
    remapped_model2_predictions = ensemble_learning.remap_predictions(model2_predictions, class_id_mapping)

    print("\n Ensemble Learning: Predictions:")
    # IMPORTANT!
    # Always send model1_predictions (bigger model) as the first arg
    # Always send remapped_model2_predictions (bigger model) as the second arg
    predictions = ensemble_learning.ensemble_predictions(model1_predictions, remapped_model2_predictions)

    # Validation
    args = dict(model=r"D:\AACV\Ensemble Learning\ensemble models\model1.pt",
                data='D:\AACV\Ensemble Learning\data\idd_data.yaml')
    print(args)
    print(type(args))
    validator = DetectionValidator(args=args, custom_predictions=predictions)
    results = validator()

    print("\n Ensemble Learning: Val Metrics")
    print(results)

    # TODO: check the confidence and threshold values for train, predict, val