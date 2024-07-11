from ultralytics import YOLO

# Load the pretrained YOLOv8 base model
model = YOLO('C:/Users/kaiwe/Documents/Master/Semester 3/AACV/YOLOv8/Hierarchical_classification/best.pt')

# Run validation on a set specified as 'val' argument
metrics = model.val(data='config.yaml')

print(metrics.results_dict)