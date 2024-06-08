from ultralytics import YOLO
from yolo_cbam import YOLOv8_CBAM  # Assuming you have imported your YOLOv8_CBAM class correctly
import multiprocessing

def main():
    # Load pretrained model
    model = YOLO("yolov8n.pt")

    # Initialize your customized CBAM model
    cbam_model = YOLOv8_CBAM(cbam_channels=[128, 256])  # Assuming this is how you initialize your CBAM model

    # Set the backbone of the YOLOv8 model to your customized CBAM model
    model.model.backbone = cbam_model

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

