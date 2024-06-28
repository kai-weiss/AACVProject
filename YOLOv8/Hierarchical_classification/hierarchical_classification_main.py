import yaml
from hierarchical_classification import *

threshold = 0.75

# Paths for the prediction files
predictions_file = "predictions.json"
new_predictions_file = "new_predictions.json"

save_new_predictions = True
plot_new_images = False

# check config file
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    img_path = config['path'] + '/' + config['test'] + '/images'
    species_mapping = config['species_mapping']
    family_mapping = config['family_mapping']

new_predictions = run_hierarchical_classification(predictions_file=predictions_file,
                                                  threshold=threshold,
                                                  species_mapping=species_mapping,
                                                  family_mapping=family_mapping)

# Save the modified predictions to a new JSON file
if save_new_predictions:
    with open(new_predictions_file, 'w') as f:
        json.dump(new_predictions, f, indent=4)

if plot_new_images:
    plot_with_new_predictions(img_path, new_predictions, species_mapping)
