import os
import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model for person detection
person_model = YOLO("C:/Users/asus/runs/detect/train33/weights/best_saved_model")

# Define paths
base_image_dir = 'datasets/output/images'  # This should have train, val, test subdirectories
base_output_dir = 'datasets/cropped/images'
bbox_info_dir = os.path.join(base_output_dir, 'bbox_info')
os.makedirs(bbox_info_dir, exist_ok=True)

# Process each subdirectory (train, val, test)
for subdir in ['train', 'val', 'test']:
    image_dir = os.path.join(base_image_dir, subdir)
    output_dir = os.path.join(base_output_dir, subdir)
    bbox_info_file = os.path.join(bbox_info_dir, f'bbox_info_{subdir}.txt')

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open a file to store bounding box information
    with open(bbox_info_file, 'w') as f:
        # Loop through images and detect persons
        for img_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_name)
            img = cv2.imread(img_path)
            results = person_model(img)

            # Extract boxes, confidence scores, and class indices
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            confs = results[0].boxes.conf.cpu().numpy()  # Confidence scores
            classes = results[0].boxes.cls.cpu().numpy()  # Class indices

            # Loop through detected persons and save cropped images
            for i, (bbox, conf, cls) in enumerate(zip(boxes, confs, classes)):
                if int(cls) == 0:  # Assuming '0' is the class index for 'person'
                    x1, y1, x2, y2 = map(int, bbox)
                    cropped_img = img[y1:y2, x1:x2]
                    crop_img_name = f'{img_name.split(".")[0]}_person_{i}.jpg'
                    cv2.imwrite(os.path.join(output_dir, crop_img_name), cropped_img)

                    # Save the bounding box information in the file
                    f.write(f'{crop_img_name},{img_name},{x1},{y1},{x2},{y2}\n')

    print(f"Cropped images and bounding box information saved successfully for {subdir}.")
