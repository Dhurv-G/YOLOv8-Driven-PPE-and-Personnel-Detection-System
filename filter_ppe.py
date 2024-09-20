import os

# Directory paths
labels_dir = 'datasets/output/labels'
output_labels_dir = 'datasets/output/ppe_labels'

# Ensure the output directory exists
os.makedirs(output_labels_dir, exist_ok=True)

# List of PPE Class IDs (based on your original annotation format)
PPE_CLASS_IDS = {
    1: 'hard-hat',
    2: 'gloves',
    3: 'mask',
    4: 'glasses',
    5: 'boots',
    6: 'vest',
    7: 'ppe-suit',
    8: 'ear-protector',
    9: 'safety-harness'
}

def filter_labels(src_file, dest_file):
    with open(src_file, 'r') as infile, open(dest_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            class_id = int(parts[0])
            if class_id in PPE_CLASS_IDS:
                outfile.write(line)

    # Remove the file if it's empty after filtering
    if os.stat(dest_file).st_size == 0:
        os.remove(dest_file)

def process_directory(subdir):
    src_path = os.path.join(labels_dir, subdir)
    dest_path = os.path.join(output_labels_dir, subdir)
    os.makedirs(dest_path, exist_ok=True)

    for filename in os.listdir(src_path):
        if filename.endswith('.txt'):
            src_file = os.path.join(src_path, filename)
            dest_file = os.path.join(dest_path, filename)
            filter_labels(src_file, dest_file)

# Process each subdirectory
for subdir in ['train', 'val', 'test']:
    process_directory(subdir)
