import os
import xml.etree.ElementTree as ET
import yaml
from PIL import Image

# Paths to your dataset
annotations_path = 'path/to/annotations/'
images_path = 'path/to/images/'
output_path = 'path/to/output/'

# Function to convert XML annotations (from IDD) to YOLO format
def convert_to_yolo(xml_file, img_width, img_height, class_name_to_id):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_annotations = []

    for obj in root.findall('object'):
        class_id = obj.find('name').text
        if class_id not in class_name_to_id:
            continue

        class_id = class_name_to_id[class_id]

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return yolo_annotations


# Function to read the image dimensions
def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.width, img.height


# Load class mappings from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    class_name_to_id = config['names']

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Process each annotation file
for xml_file in os.listdir(annotations_path):
    if xml_file.endswith('.xml'):
        img_file = os.path.splitext(xml_file)[0] + '.jpg'
        img_path = os.path.join(images_path, img_file)

        if not os.path.exists(img_path):
            continue

        img_width, img_height = get_image_dimensions(img_path)
        yolo_annotations = convert_to_yolo(os.path.join(annotations_path, xml_file), img_width, img_height,
                                           class_name_to_id)

        # Save the YOLO annotations to a txt file
        output_file = os.path.join(output_path, os.path.splitext(xml_file)[0] + '.txt')
        with open(output_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))
