import yaml
from hierarchical_classification import *

threshold = 0.75

# Paths for the prediction files
predictions_file = "predictions.json"
new_predictions_file = "new_predictions.json"

calc_hierarchical_classification = True
save_new_predictions = True
plot_new_images = True
print_results = False  # Only set true if you do have the new_predictions.json file


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
    with open(new_predictions_file, 'r') as f:
        new_predictions = json.load(f)


# Save the modified predictions to a new JSON file
if save_new_predictions and new_predictions is not None:
    with open(new_predictions_file, 'w') as f:
        json.dump(new_predictions, f, indent=4)


if plot_new_images and new_predictions is not None:
    unique_image_ids = unique_imagesids_from_predictions(new_predictions)
    random_unique_image_ids = unique_image_ids
    random.shuffle(random_unique_image_ids)
    random_unique_image_ids = ["sideRight_BLR-2018-05-09_12-50-30_sideRight_0009780.jpg", "frontFar_BLR-2018-04-19_17-16-55_frontFar_001464_r.jpg", "sideLeft_BLR-2018-05-08_10-11-11_sideLeft_000936_r.jpg"]
    plot_n_images_from_imageids_list(random_unique_image_ids, new_predictions, species_mapping, img_path, label_path)


