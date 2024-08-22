import yaml

from Hierarchical_classification.hierarchical_classification_val import validate_hierarchical_classification
from hierarchical_classification import *


# Hierarchical threshold
threshold = 0.75

# Paths for the prediction files
predictions_file = "predictions.json"
new_predictions_file = "universal_hierarchical_predictions.json"


# Set True to run the according methods
calc_hierarchical_classification = True
save_new_predictions = True
plot_new_images = False
validate_hier_classification = False


# Images that you want to plot over
unique_image_ids = ["sideRight_BLR-2018-05-09_12-50-30_sideRight_0009780.jpg",
                    "frontFar_BLR-2018-04-19_17-16-55_frontFar_001464_r.jpg",
                    "sideLeft_BLR-2018-05-08_10-11-11_sideLeft_000936_r.jpg"]


# check config file
with open('../YOLOv8/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    img_path = config['path'] + '/' + config['val'] + '/images'
    label_path = config['path'] + '/' + config['val'] + '/labels'
    species_mapping = config['species_mapping']
    family_mapping = config['family_mapping']


# Calculation
if calc_hierarchical_classification:
    new_predictions = run_hierarchical_classification(predictions_file=predictions_file,
                                                      hier_threshold=threshold,
                                                      species_mapping=species_mapping,
                                                      family_mapping=family_mapping,
                                                      universal=True)
else:
    with open(new_predictions_file, 'r') as f:
        new_predictions = json.load(f)


# Save the modified predictions to a new JSON file
if save_new_predictions and new_predictions is not None:
    with open(new_predictions_file, 'w') as f:
        json.dump(new_predictions, f, indent=4)


# Plotting
if plot_new_images and new_predictions is not None:
    plot_n_images_from_imageids_list(unique_image_ids, new_predictions, species_mapping, img_path, label_path)


# Validation
if validate_hier_classification and new_predictions is not None:
    results = validate_hierarchical_classification(predictions_file=predictions_file,
                                                   label_path=label_path)
    print(results)


