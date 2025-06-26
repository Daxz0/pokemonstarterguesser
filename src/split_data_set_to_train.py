import random
import shutil
from pathlib import Path

import Constants

dataset_root = Constants.OBJECT_DETECTION_DATASET_PATH
images_dir = Constants.DETECTION_IMAGES_PATH
labels_dir = Constants.DETECTION_LABELS_PATH

# Output folders
train_images = images_dir / "train"
val_images = images_dir / "val"
train_labels = labels_dir / "train"
val_labels = labels_dir / "val"

image_files = list(images_dir.glob("*.png"))

random.shuffle(image_files)
split_ratio = 0.8  # 80% train
split_idx = int(len(image_files) * split_ratio)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def copy_pairs(files, image_dest, label_dest):
    for image_path in files:
        label_path = labels_dir / (image_path.stem + ".txt")
        if label_path.exists():
            shutil.copy(image_path, image_dest / image_path.name)
            shutil.copy(label_path, label_dest / label_path.name)
        else:
            print(f"No label for {image_path.name}, skipping.")

# Copy files
copy_pairs(train_files, train_images, train_labels)
copy_pairs(val_files, val_images, val_labels)

print(f"Split complete! {len(train_files)} train and {len(val_files)} val images.")
