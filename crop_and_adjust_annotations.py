import os
from ultralytics import YOLO
from PIL import Image

def adjust_annotation_for_cropped_image(original_txt_path, cropped_img_bbox, cropped_img_path, output_dir):
    try:
        with open(original_txt_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"File not found: {original_txt_path}")
        return

    x1_crop, y1_crop, x2_crop, y2_crop = cropped_img_bbox
    cropped_width = x2_crop - x1_crop
    cropped_height = y2_crop - y1_crop
    new_annotations = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            print(f"Skipping invalid line in {original_txt_path}: {line}")
            continue
        
        cls = int(parts[0])
        if cls == 0:  # Exclude class with ID 0
            continue
        
        x_center, y_center, width, height = map(float, parts[1:])
        x_center = int(x_center * cropped_width)
        y_center = int(y_center * cropped_height)
        width = int(width * cropped_width)
        height = int(height * cropped_height)

        x1_original = x_center - width // 2
        y1_original = y_center - height // 2
        x2_original = x_center + width // 2
        y2_original = y_center + height // 2

        # Adjust bounding box coordinates relative to the cropped image
        adjusted_x1 = max(x1_original - x1_crop, 0)
        adjusted_y1 = max(y1_original - y1_crop, 0)
        adjusted_x2 = min(x2_original - x1_crop, cropped_width)
        adjusted_y2 = min(y2_original - y1_crop, cropped_height)

        if adjusted_x2 > adjusted_x1 and adjusted_y2 > adjusted_y1:
            # Convert to YOLO format
            new_x_center = (adjusted_x1 + adjusted_x2) / 2 / cropped_width
            new_y_center = (adjusted_y1 + adjusted_y2) / 2 / cropped_height
            new_width = (adjusted_x2 - adjusted_x1) / cropped_width
            new_height = (adjusted_y2 - adjusted_y1) / cropped_height
            new_annotations.append(f"{cls} {new_x_center} {new_y_center} {new_width} {new_height}\n")
        else:
            print(f"Invalid annotation for cropped image: {cropped_img_path} - Bounding box: {(adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2)}")

    if not new_annotations:
        print(f"No valid annotations for cropped image: {cropped_img_path}")
        # Optionally, remove the cropped image if there are no valid annotations
        if os.path.exists(cropped_img_path):
            os.remove(cropped_img_path)
        return

    # Save the new annotation file
    output_txt_path = os.path.join(output_dir, os.path.basename(cropped_img_path).replace('.jpg', '.txt'))
    with open(output_txt_path, 'w') as file:
        file.writelines(new_annotations)
    print(f"Saved annotations for {cropped_img_path} to {output_txt_path}")

def process_detections(detections, original_img_path, output_img_dir, output_ann_dir, base_annotation_dir):
    original_img = Image.open(original_img_path)
    
    # Get the subdirectory (e.g., 'test', 'train', 'val') from the image path
    subdir = os.path.basename(os.path.dirname(original_img_path))
    
    # Construct the correct path to the annotation file
    original_txt_path = os.path.join(base_annotation_dir, subdir, os.path.basename(original_img_path).replace('.jpg', '.txt'))

    # Debugging line to print the annotation path
    print(f"Looking for annotation file: {original_txt_path}")

    # The rest of your existing code...
    for i, (x1, y1, x2, y2) in enumerate(detections):
        cropped_img = original_img.crop((x1, y1, x2, y2))
        cropped_img_name = os.path.basename(original_img_path).replace('.jpg', f'_crop_{i}.jpg')
        cropped_img_path = os.path.join(output_img_dir, cropped_img_name)
        cropped_img.save(cropped_img_path)

        # Adjust annotations for the cropped image
        adjust_annotation_for_cropped_image(original_txt_path, (x1, y1, x2, y2), cropped_img_path, output_ann_dir)

# Load YOLOv8 model
model = YOLO("C:/Users/asus/runs/detect/train33/weights/best_saved_model")  # Replace with your custom model if needed

# Define paths
base_img_dir = 'datasets/output/images'
base_annotation_dir = 'datasets/output/labels'  # Path where original annotations are stored
base_output_img_dir = 'datasets/cropped/images'
base_output_ann_dir = 'datasets/cropped/labels'

# Create output directories
os.makedirs(base_output_img_dir, exist_ok=True)
os.makedirs(base_output_ann_dir, exist_ok=True)

# Process each subdirectory (train, val, test)
for subdir in ['train', 'val', 'test']:
    img_dir = os.path.join(base_img_dir, subdir)
    output_img_dir = os.path.join(base_output_img_dir, subdir)
    output_ann_dir = os.path.join(base_output_ann_dir, subdir)
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_ann_dir, exist_ok=True)

    for img_name in os.listdir(img_dir):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(img_dir, img_name)
            
            # Perform inference
            results = model(img_path)
            
            # Extract bounding boxes from results
            detections = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])  # Convert to int
                    detections.append((x1, y1, x2, y2))
            
            # Process detections (crop and adjust annotations)
            process_detections(detections, img_path, output_img_dir, output_ann_dir, base_annotation_dir)

    print(f"Processed images and annotations for {subdir}.")

print("All processing complete.")
