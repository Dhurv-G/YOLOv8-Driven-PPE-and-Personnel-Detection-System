import os

# Directory paths
labels_dir = 'datasets/output/labels'  # Update with your actual path
output_labels_dir = 'datasets/output/peron_only_labels'  # Directory for filtered labels

# Ensure the output directory exists
os.makedirs(output_labels_dir, exist_ok=True)

# Class ID for person (assuming class ID is 0, change if needed)
PERSON_CLASS_ID = 0

def filter_labels(src_file, dest_file):
    with open(src_file, 'r') as infile, open(dest_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if int(parts[0]) == PERSON_CLASS_ID:
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
