import os
import Constants
import random
import glob
import cv2
from PIL import Image
from rembg import remove
from conversion_between_yolo_coordinates import to_yolo_format

# Load background once
background = Image.open("object_detection_background.jpg")
#background = cv2.resize(background, (2000, 2000), interpolation=cv2.INTER_AREA)
background = background.resize((2000, 2000))
max_width, max_height = background.size

# Get all valid image file paths
print("Classification Data Images Path:", Constants.CLASSIFICATION_DATA_PATH)
print("Entries:", os.listdir(Constants.CLASSIFICATION_DATA_PATH))

files = glob.glob(os.path.join(Constants.CLASSIFICATION_DATA_PATH, "**", "*.*"), recursive=True)
files = [f for f in files if os.path.isfile(f) and f.lower().endswith((".png", ".jpg", ".jpeg"))]

print("Found", len(files), "image files to process.")

# Loop through each file safely
for counter, image_path in enumerate(files):
    try:
        print(f"Processing image: {image_path}")
        image = Image.open(image_path).convert("RGBA")  # Convert to RGBA for transparency
        image_width, image_height = image.size

        #cv2.resize(image, (image_width / 1.5, image_height / 1.5), interpolation=cv2.INTER_AREA)

        image_no_bg = remove(image)  # Fix: Use `image`, not `input`
        image_no_bg_width, image_no_bg_height = image_no_bg.size

        # Ensure position fits within background
        random_X = random.randint(0, max_width - image_no_bg_width)
        random_y = random.randint(0, max_height - image_no_bg_height)
        position = (random_X, random_y)

        combined_image = background.copy()
        combined_image.paste(image_no_bg, position, image_no_bg)  # Use mask

        output_path = os.path.join('C:\\Data\\Anish\\Github\\pokemonstarterguesser\\object_detection_dataset\\images',f"{counter}.png")
        combined_image.save(output_path)
        print(f"Successfully saved image: {output_path}")

        x_min = random_X
        y_min = random_y
        x_max = random_X + image_no_bg_height
        y_max = random_y + image_no_bg_height

        # Normalize for YOLO format
        x_center, y_center, width, height = to_yolo_format(x_min, y_min, x_max, y_max, max_width, max_height)

        # Save label
        label_filename = f"{counter}.txt"
        label_path = os.path.join('C:\\Data\\Anish\\Github\\pokemonstarterguesser\\object_detection_dataset\\labels', label_filename)
        with open(label_path, "w") as f:
            f.write(f"{0} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        print(f"Label saved: {label_path}")

    except Exception as e:
        print(f"Failed to process {image_path}: {e}")