from ultralytics import YOLO

# new model from scratch
# model = YOLO("yolov8n.yaml")

# pretrained model
model = YOLO("yolov8n.pt")

# Train model
results = model.train(data="config.yaml", epochs=300)
