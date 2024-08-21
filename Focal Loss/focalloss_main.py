import ultralytics
from ultralytics import YOLO

def main():
    # Load the model
    model = YOLO("yolov8s.pt")
    
    # Train the model
    results = model.train(data="/data/idd_data.yaml", epochs=200)
    
    # Validate the model and print metrics
    metrics = model.val()
    print("Validation metrics:")
    print(metrics)

if __name__ == "__main__":
    main()
