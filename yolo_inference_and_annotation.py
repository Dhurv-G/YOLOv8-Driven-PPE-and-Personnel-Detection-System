import os
from ultralytics import YOLO

# Load your trained model
model_path = 'C:/Users/asus/runs/detect/train54/weights/best.pt'
model = YOLO(model_path)

# Define your subdirectories for train, val, and test
sub_dirs = ['train', 'val', 'test']
image_base_dir = 'datasets/cropped/images'
label_base_dir = 'datasets/cropped/labels'

# Ensure label directories exist
for sub_dir in sub_dirs:
    os.makedirs(os.path.join(label_base_dir, sub_dir), exist_ok=True)

# Iterate over each subdirectory (train, val, test)
for sub_dir in sub_dirs:
    image_dir = os.path.join(image_base_dir, sub_dir)
    output_dir = os.path.join(label_base_dir, sub_dir)

    # Process each image in the directory
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)

        # Run inference on the image
        results = model(img_path)

        # Save the results
        for i, result in enumerate(results):
            # Extract bounding boxes, classes
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            # Get the image size to scale bounding boxes if necessary
            image_size = result.orig_shape  # (height, width)

            # Prepare YOLO format annotations
            annotations = []
            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                x_center = x1 + width / 2
                y_center = y1 + height / 2

                # Normalize coordinates (YOLO format)
                x_center /= image_size[1]
                y_center /= image_size[0]
                width /= image_size[1]
                height /= image_size[0]

                # Append to annotations without the confidence score
                annotations.append(f"{int(cls)} {x_center} {y_center} {width} {height}")

            # Save the annotations to the corresponding text file
            label_file = img_file.replace('.jpg', '.txt')
            label_path = os.path.join(output_dir, label_file)
            with open(label_path, 'w') as f:
                f.write('\n'.join(annotations))

print("Inference and label generation complete.")
