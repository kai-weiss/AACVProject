from ultralytics import YOLO
import yaml

# Load the trained model
model = YOLO('best.pt')

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    source = config['path'] + '/' + config['test'] + '/images'

results = model.predict(source, save=True)

