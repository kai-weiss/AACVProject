import yaml
from predict_with_extraction import *

# Path to the trained model
model_path = 'best.pt'

# check config file
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    img_path = config['path'] + '/' + config['val'] + '/images'
    category_mapping = config['names']

# run inference
results = run_predict(input_path=img_path,
                      model_path=model_path,
                      score_threshold=0.25,  # default nms conf
                      iou_threshold=0.7,  # default iou
                      save_image=False,
                      save_json=True,
                      category_mapping=category_mapping,
                      softmax_temperature_value=3,
                      agnostic=False,
                      )

for result in results:
    print("\n")
    print("Bounding Box :" + str(result['bbox']))
    print("Logits :" + str(result['logits']))
    print("Activations :" + str(result['activations']))
    print("\n")

# plot_image(img_path, results)
