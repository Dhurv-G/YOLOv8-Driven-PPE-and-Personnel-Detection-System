# YOLO Object Detection for PPE

## Model Versions
- **train33**: Model for Person Detection
- **train54**: Model for PPE Detection
- **train59**: Model for Cropped Detection
- **annotated_images**: Set of images with drawn bounding boxes

## Introduction
This repository showcases the implementation and evaluation of a YOLO-based object detection system tailored for detecting personal protective equipment (PPE). The project encompasses the conversion of PascalVOC annotations to YOLOv8 format, the training of YOLOv8 models on cropped images, and the execution of inference with visualizations.

## Objectives
- Convert PascalVOC annotations into YOLOv8 format.
- Train a YOLOv8 model on cropped PPE images for improved accuracy.
- Conduct inference and visualize results with annotated bounding boxes for clarity.

## Methodology

1. **Annotation Conversion**
   - **Script**: `PascalVOC_to_yolo.py`
   - **Purpose**: Converts PascalVOC annotations into the YOLOv8 format, ensuring correct adjustments for cropped images.

2. **Data Preparation**
   - **Script**: `crop_and_adjust_annotations.py`
   - **Purpose**: Adjusts annotations for cropping and resizing, generating YOLO-format labels for training.

3. **Model Training**
   - **Script**: `PPE_detection.py`
   - **Purpose**: Trains the YOLOv8 model on cropped PPE images, managing data loading, model configuration, training, evaluation, and model saving.

4. **Inference and Visualization**
   - **Script**: `yolo_inference_and_annotation.py`
   - **Purpose**: Performs inference on test images using the trained models and visualizes results by drawing bounding boxes and labels. Bounding boxes are rendered using OpenCV, utilizing distinct colors for various classes, including cyan, orange, white, yellow, and yellowish-green.

## Results
- **Training Performance**: Key metrics such as precision, recall, and mean Average Precision (mAP) are provided to evaluate model effectiveness.
- **Inference Results**: Examples illustrate detected PPE with visualized bounding boxes and labels on test images.

## Challenges and Learnings
- **Data Conversion**: Encountered and addressed challenges related to converting annotations and adjusting for cropped images.
- **Model Training**: Gained valuable insights from training and fine-tuning hyperparameters to optimize performance.
- **Inference**: Observed results during inference execution and the visualization of detection outputs.

## Conclusion
This project effectively demonstrates the full workflow of converting annotations, training a YOLOv8 model, and executing object detection for PPE. The final model provides accurate detection results, and the accompanying scripts facilitate a comprehensive approach from data preparation to inference.

## Files
- `convert_voc_to_yolo.py`: Converts PascalVOC annotations to YOLOv8 format.
- `crop_and_adjust_annotations.py`: Prepares annotations for cropped images.
- `PPE_detection.py`: Trains the YOLOv8 model on cropped PPE images.
- `yolo_inference_and_annotation.py`: Performs inference and visualizes results.
- `weights/`: Directory containing the trained model weights.

## How to Use

### Annotation Conversion:
```bash
python convert_voc_to_yolo.py <input_dir> <output_dir>
