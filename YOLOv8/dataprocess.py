import os
import random
import shutil
from collections import defaultdict

def create_class_balanced_subset(original_annotations_dir, original_images_dir, subset_annotations_dir, subset_images_dir, subset_percentage, min_samples_per_class=10):
    # Create subset directories if they don't exist
    if subset_annotations_dir:
        os.makedirs(subset_annotations_dir, exist_ok=True)
    if subset_images_dir:
        os.makedirs(subset_images_dir, exist_ok=True)

    # Dictionary to hold files by class
    class_to_files = defaultdict(list)

    # List all annotation files in the original directory
    if original_annotations_dir:
        annotation_files = [f for f in os.listdir(original_annotations_dir) if f.endswith('.txt')]
    else:
        annotation_files = []

    # Read all annotations and categorize by class
    for file_name in annotation_files:
        file_path = os.path.join(original_annotations_dir, file_name)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if not lines:  # Skip empty annotation files
                continue
            class_id = int(lines[0].split()[0])  # Get class ID from the first line
            class_to_files[class_id].append(file_name)

    # Create subset list
    subset_files = set()
    for class_id, files in class_to_files.items():
        # Number of samples to take from this class
        num_samples = max(int(len(files) * subset_percentage), min_samples_per_class)
        selected_files = random.sample(files, min(num_samples, len(files)))
        subset_files.update(selected_files)

    # Copy selected annotation files to the subset directory
    if subset_annotations_dir:
        for file_name in subset_files:
            original_annotation_path = os.path.join(original_annotations_dir, file_name)
            subset_annotation_path = os.path.join(subset_annotations_dir, file_name)
            shutil.copyfile(original_annotation_path, subset_annotation_path)

    # Copy corresponding images to the subset directory if they exist
    if original_images_dir and subset_images_dir and os.path.exists(original_images_dir):
        for file_name in subset_files:
            image_name = file_name.replace('.txt', '.jpg')  # Adjust this if your images have a different extension
            original_image_path = os.path.join(original_images_dir, image_name)
            subset_image_path = os.path.join(subset_images_dir, image_name)
            if os.path.exists(original_image_path):
                shutil.copyfile(original_image_path, subset_image_path)

def create_subset_images(original_images_dir, subset_images_dir, subset_percentage):
    # Create subset directory if it doesn't exist
    os.makedirs(subset_images_dir, exist_ok=True)

    # List all image files in the original directory
    image_files = [f for f in os.listdir(original_images_dir) if f.endswith('.jpg')]

    # Calculate the number of images to select for the subset
    num_images_subset = int(len(image_files) * subset_percentage)

    # Select a random subset of images
    selected_images = random.sample(image_files, num_images_subset)

    # Copy selected images to the subset directory
    for image_name in selected_images:
        original_image_path = os.path.join(original_images_dir, image_name)
        subset_image_path = os.path.join(subset_images_dir, image_name)
        shutil.copyfile(original_image_path, subset_image_path)

# Define paths for train, val, and test sets
original_train_annotations_dir = 'D:/Coding/Uni/traffic3/IDD10_converted/images/train/labels'
original_train_images_dir = 'D:/Coding/Uni/traffic3/IDD10_converted/images/train/images'
subset_train_annotations_dir = 'D:/Coding/Uni/traffic3/shrunk/train/labels'
subset_train_images_dir = 'D:/Coding/Uni/traffic3/shrunk/train/images'

original_val_annotations_dir = 'D:/Coding/Uni/traffic3/IDD10_converted/images/val/labels'
original_val_images_dir = 'D:/Coding/Uni/traffic3/IDD10_converted/images/val/images'
subset_val_annotations_dir = 'D:/Coding/Uni/traffic3/shrunk/val/labels'
subset_val_images_dir = 'D:/Coding/Uni/traffic3/shrunk/val/images'

original_test_annotations_dir = 'D:/Coding/Uni/traffic3/IDD10_converted/images/test/labels'
original_test_images_dir = 'D:/Coding/Uni/traffic3/IDD10_converted/images/test/images'
subset_test_annotations_dir = 'D:/Coding/Uni/traffic3/shrunk/test/labels'
subset_test_images_dir = 'D:/Coding/Uni/traffic3/shrunk/test/images'

# Specify the percentage of samples to include in the subset
subset_percentage = 0.05

# Create subsets for train, val, and test sets
create_class_balanced_subset(original_train_annotations_dir, original_train_images_dir, subset_train_annotations_dir, subset_train_images_dir, subset_percentage)
create_class_balanced_subset(original_val_annotations_dir, original_val_images_dir, subset_val_annotations_dir, subset_val_images_dir, subset_percentage)
create_subset_images(original_test_images_dir, subset_test_images_dir, subset_percentage)
