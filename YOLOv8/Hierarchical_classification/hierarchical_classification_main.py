import yaml
from hierarchical_classification import *

threshold = 0.75

# Paths for the prediction files
predictions_file = "predictions.json"
new_predictions_file = "new_predictions.json"

calc_hierarchical_classification = False
save_new_predictions = False
plot_new_images = False
print_results = True  # Only set true if you do have the new_predictions.json file


# check config file
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    img_path = config['path'] + '/' + config['val'] + '/images'
    label_path = config['path'] + '/' + config['val'] + '/labels'
    species_mapping = config['species_mapping']
    family_mapping = config['family_mapping']

if calc_hierarchical_classification:
    new_predictions = run_hierarchical_classification(predictions_file=predictions_file,
                                                      threshold=threshold,
                                                      species_mapping=species_mapping,
                                                      family_mapping=family_mapping)
else:
    new_predictions = None


# Save the modified predictions to a new JSON file
if save_new_predictions and new_predictions is not None:
    with open(new_predictions_file, 'w') as f:
        json.dump(new_predictions, f, indent=4)


if plot_new_images and new_predictions is not None:
    plot_with_new_predictions(img_path, new_predictions, species_mapping)


if print_results and label_path != '':
    predict_with_hierarchical_classification(new_predictions_file, label_path)
elif label_path == '':
    print('No label directory!')
