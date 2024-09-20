YOLO Object Detection for PPE

train33-person detection model
train54-ppe detection model
train59-cropped detection model
annotated_images-images with bounding boxes drawn on them

Introduction
This repository contains the implementation and evaluation of a YOLO-based object detection system for detecting personal protective equipment (PPE). The project involves converting PascalVOC annotations to YOLOv8 format, training YOLOv8 models on cropped images, and performing inference with visualization.

Objective
Convert PascalVOC annotations to YOLOv8 format.
Train a YOLOv8 model on cropped PPE images.
Perform inference and visualize results with annotated bounding boxes.
Methodology
1. Annotation Conversion
Script: PascalVOC_to_yolo.py
Purpose: Converts PascalVOC annotations into YOLOv8 format, ensuring that annotations are correctly adjusted for the new cropped images.
2. Data Preparation
Script: crop_and_adjust_annotations.py
Purpose: Adjusts annotations for image cropping and resizing, generating YOLO-format labels for training on cropped images.
3. Model Training
Script: PPE_detection.py
Purpose: Trains the YOLOv8 model on cropped PPE images. This script handles data loading, model configuration, training, and evaluation, and saves the trained model.
4. Inference and Visualization
Script: yolo_inference_and_annotation.py
Purpose: Performs inference on test images using the trained models and visualizes results by drawing bounding boxes and labels. Bounding boxes are drawn using OpenCV functions, and colors for different classes include cyan, orange, white, yellow, and yellowish-green.
Results
Training Performance: Metrics including precision, recall, and mean Average Precision (mAP) are provided to assess model effectiveness.
Inference Results: Examples of detected PPE with visualized bounding boxes and labels on test images.
Challenges and Learnings
Data Conversion: Challenges with converting annotations and adjusting for cropped images.
Model Training: Insights gained from training and fine-tuning hyperparameters.
Inference: Observations from running the inference script and visualizing results.
Conclusion
The project successfully demonstrated the process of converting annotations, training a YOLOv8 model, and performing object detection for PPE. The final model provides accurate detection results, and the scripts facilitate a comprehensive workflow from data preparation to inference.

Files
convert_voc_to_yolo.py: Converts PascalVOC annotations to YOLOv8 format.
crop_and_adjust_annotations.py: Prepares annotations for cropped images.
PPE_detection.py: Trains the YOLOv8 model on cropped PPE images.
yolo_inference_and_annotation.py: Performs inference and visualizes results.
weights/: Directory containing trained model weights.
How to Use
Annotation Conversion:

python convert_voc_to_yolo.py <input_dir> <output_dir>

Data Preparation: Ensure annotations are adjusted using crop_and_adjust_annotations.py.

Model Training:

python PPE_detection.py

Inference and Visualization:

python yolo_inference_and_annotation.py --input_dir <input_dir> --output_dir <output_dir> --person_det_mode