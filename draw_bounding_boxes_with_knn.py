import cv2
from ultralytics import YOLO
from k_nearest_neighbors import KNearestNeighbors  # Your custom class
import init
from PIL import Image
import numpy as np

def draw_bounding_boxes_with_knn(image_path, knn_model_path='pokemon_classifier_model.pkl', yolo_model_path='runs/detect/train6/weights/best.pt'):
    # Load models
    model = YOLO(yolo_model_path)
    knn_model = KNearestNeighbors.load_model(knn_model_path)

    # Load image
    image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run YOLO inference
    results = model(image_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # Crop the detected region
        cropped = img_rgb[y1:y2, x1:x2]
        if cropped.size == 0:
            continue  # Skip invalid crops

        # Resize and flatten for KNN prediction
        img_pil = Image.fromarray(cropped).resize((32, 32)).convert('RGB')
        flat = np.array(img_pil).flatten().reshape(1, -1)
        label = knn_model.predict(flat)[0]

        # Draw rectangle and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # Save output
    output_path = "output_with_knn_labels.jpg"
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")