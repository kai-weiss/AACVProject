from ultralytics import YOLO

# new model from scratch
# model = YOLO("yolov8n.yaml")

# pretrained model
model = YOLO("yolov8n.pt")

# Train model
results = model.train(data="config.yaml", epochs=300, patience=50)

# Evaluate model performance
#results = model.val()

# Predict on an image
#results = model("?")


#success = model.export(format="onnx")
