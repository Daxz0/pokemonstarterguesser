from ultralytics import YOLO

# Train a YOLOv8 model
model = YOLO('yolov8n.pt')  # Load pretrained model

# Train using your data.yaml
model.train(data='data.yaml', epochs=50, imgsz=640)