import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import math

def split_dataset(image_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-9), "Ratios must sum to 1."

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    
    # Create directories for images and labels
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'

    train_dir = images_dir / 'train'
    val_dir = images_dir / 'val'
    test_dir = images_dir / 'test'

    train_labels_dir = labels_dir / 'train'
    val_labels_dir = labels_dir / 'val'
    test_labels_dir = labels_dir / 'test'

    # Create necessary directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    # Get and shuffle images
    images = list(image_dir.glob('*.jpg'))
    random.shuffle(images)
    
    train_count = int(len(images) * train_ratio)
    val_count = int(len(images) * val_ratio)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Copy images to the respective directories
    for img in train_images:
        shutil.copy(img, train_dir / img.name)

    for img in val_images:
        shutil.copy(img, val_dir / img.name)

    for img in test_images:
        shutil.copy(img, test_dir / img.name)

    print(f"Train set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")
    print(f"Test set: {len(test_images)} images")

def convert_voc_to_yolo(voc_dir, output_dir, classes, splits=('train', 'val', 'test')):
    for split in splits:
        split_images_dir = output_dir / 'images' / split
        annotation_dir = output_dir / 'labels' / split
        annotation_dir.mkdir(parents=True, exist_ok=True)

        for filename in split_images_dir.glob('*.jpg'):
            xml_path = voc_dir / (filename.stem + '.xml')
            if not xml_path.exists():
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()
            width = int(root.find('size/width').text)
            height = int(root.find('size/height').text)

            yolo_annotation = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in classes:
                    continue
                class_id = classes.index(class_name)

                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                xmax = int(bbox.find('xmax').text)
                ymin = int(bbox.find('ymin').text)
                ymax = int(bbox.find('ymax').text)

                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height

                yolo_annotation.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")

            with open(annotation_dir / (filename.stem + '.txt'), 'w') as f:
                f.write('\n'.join(yolo_annotation))

def parse_args():
    parser = argparse.ArgumentParser(description='Split dataset and convert PascalVOC annotations to YOLO format.')
    parser.add_argument('image_dir', help='Path to the image directory')
    parser.add_argument('annotation_dir', help='Path to the PascalVOC annotations directory')
    parser.add_argument('output_dir', help='Path to the output directory')
    parser.add_argument('classes_file', help='Path to the classes.txt file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(args.classes_file, 'r') as f:
        classes = f.read().strip().split()

    split_dataset(args.image_dir, args.output_dir)
    convert_voc_to_yolo(Path(args.annotation_dir), Path(args.output_dir), classes)
