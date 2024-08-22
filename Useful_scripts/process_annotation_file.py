import os

# Mapping of original class indices to new class indices
class_mapping = {
    0: 1,  # car
    1: 1,  # bus
    2: 1,  # autorickshaw
    3: 1,  # truck
    4: 0,  # motorcycle
    5: 3,  # rider
    6: 3,  # person
    7: 0,  # bicycle
    8: 2,  # traffic sign
    9: 2  # traffic light
}

#class_mapping = {
#    10: 0,  # car
#    11: 1,  # bus
#    12: 2,  # autorickshaw
#    13: 3,  # truck
#    14: 4,  # motorcycle
#    15: 5,  # rider
#}

input_dir = r'C:\Users\kaiwe\Documents\Master\Semester 3\archive\shrunk\train\labels'

output_dir = r'C:\Users\kaiwe\Documents\Master\Semester 3\archive\shrunk\train\labels_2'

os.makedirs(output_dir, exist_ok=True)


def process_annotation_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    filtered_lines = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        if class_id in class_mapping:
            # Map the original class id to the new class id
            new_class_id = class_mapping[class_id]
            parts[0] = str(new_class_id)
            filtered_lines.append(' '.join(parts))

    with open(output_file, 'w') as file:
        for line in filtered_lines:
            file.write(line + '\n')


for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        process_annotation_file(input_file, output_file)

print('Processing completed.')
