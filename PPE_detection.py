from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(data='data_ppe.yaml', epochs=200, imgsz=640)
    model.export(format="saved_model")
    exported_model = YOLO("C:/Users/asus/runs/detect/train59/weights/best_saved_model")
    results = exported_model("C:/Users/asus/OneDrive/Documents/VScode/Syook/datasets/whole_image.jpg")
    print(results)
    
if __name__ == '__main__':
    main()