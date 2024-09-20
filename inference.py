import os
import cv2
import argparse
from ultralytics import YOLO

# Define a color map for class names
COLOR_MAP = {
    0: (255, 255, 0),    # Cyan for class 0 (Person)
    1: (0, 165, 255),    # Orange for class 1 (Hard-hat)
    2: (255, 255, 255),  # White for class 2 (Gloves)
    3: (0, 255, 255),    # Yellow for class 3 (Mask)
    4: (173, 255, 47),   # Yellowish-Green for class 4 (Glasses)
    5: (0, 0, 255),      # Red for class 5 (Boots)
    6: (255, 255, 255),  # White for other classes (default)
    7: (128, 128, 0),    # Olive for class 7 (e.g., PPE-suit)
    8: (0, 128, 128),    # Teal for class 8 (e.g., Ear-protector)
    9: (128, 128, 128)   # Gray for class 9 (e.g., Safety-harness)
}

def process_image(img_path, person_model, ppe_model, output_dir):
    img = cv2.imread(img_path)

    # Step 1: Person Detection
    person_results = person_model(img)
    persons = []

    # Extract person bounding boxes
    for person in person_results[0].boxes:
        x1, y1, x2, y2 = map(int, person.xyxy.cpu().numpy()[0])
        cropped_img = img[y1:y2, x1:x2]
        persons.append((cropped_img, x1, y1, x2, y2))

    # Step 2: PPE Detection on Cropped Persons
    for i, (cropped_img, x1, y1, x2, y2) in enumerate(persons):
        ppe_results = ppe_model(cropped_img)

        for ppe_box in ppe_results[0].boxes:
            # Get bounding box coordinates and class index
            ppe_x1, ppe_y1, ppe_x2, ppe_y2 = map(int, ppe_box.xyxy.cpu().numpy()[0])
            class_id = int(ppe_box.cls.cpu().numpy()[0])
            confidence = ppe_box.conf.cpu().numpy()[0]

            # Debugging: Print class ID and color
            print(f"Detected class ID: {class_id}, Confidence: {confidence}")

            # Map PPE bounding boxes back to full image coordinates
            full_x1 = x1 + ppe_x1
            full_y1 = y1 + ppe_y1
            full_x2 = x1 + ppe_x2
            full_y2 = y1 + ppe_y2

            # Get the class name and corresponding color
            class_name = ppe_model.names[class_id]
            color = COLOR_MAP.get(class_id, (255, 255, 255))  # Default to white if class not found
            label = f"{class_name} ({confidence:.2f})"
            
            # Draw bounding box and label with the class-specific color
            cv2.rectangle(img, (full_x1, full_y1), (full_x2, full_y2), color, 2)
            cv2.putText(img, label, (full_x1, full_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Save the annotated full image
    output_img_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_img_path, img)

def main(args):
    # Load models
    person_model = YOLO(args.person_model_path)
    ppe_model = YOLO(args.ppe_model_path)

    # Define directories
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Walk through input directory and process images
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(root, file)
                process_image(img_path, person_model, ppe_model, output_dir)

    print("Inference complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference for person and PPE detection.')
    parser.add_argument('--person_model_path', type=str, required=True, help='Path to the person detection model.')
    parser.add_argument('--ppe_model_path', type=str, required=True, help='Path to the PPE detection model.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images, including subdirectories.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save annotated output images.')

    args = parser.parse_args()
    main(args)
