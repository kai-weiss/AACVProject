from ultralytics import YOLO


print('test')
model = YOLO("yolov8n.yaml")

# Train model
results = model.train(data="config.yaml", epochs=1)

# Evaluate model performance
#results = model.val()

# Predict on an image
#results = model("?")


#success = model.export(format="onnx")
