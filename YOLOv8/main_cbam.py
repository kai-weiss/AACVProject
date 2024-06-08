from ultralytics import YOLO
import multiprocessing

def main():
    # Load pretrained model
    model = YOLO("yolov8n.pt")

    # Train model
    results = model.train(data="config.yaml", epochs=300, patience=50)

    # Evaluate model performance (if needed)
    # results = model.val()

    # Predict on an image (if needed)
    # results = model("?")

    # Export the model to ONNX format (if needed)
    # success = model.export(format="onnx")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
