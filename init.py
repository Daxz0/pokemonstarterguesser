#libraries
import Constants
import pandas as pd
import os
from collections import Counter
from PIL import Image
import numpy as np
import pandas as pd


def lower_image_resolution(
    iterations: int,
    resolution: tuple,
    images_path: str,
    output_path: str,
    # single: bool,
    label) -> None:
    
    is_labeled = label is not None
    target_path = os.path.join(output_path, label) if is_labeled else output_path

    os.makedirs(target_path, exist_ok=True)

    print(f"Converting {'Label: ' + label if is_labeled else 'Unlabeled Images'}")

    
    all_paths = os.listdir(images_path)
    if iterations < 0:
        iterations = len(all_paths)
    

    for idx, file_name in zip(range(iterations), all_paths):
        input_image_path = os.path.join(images_path, file_name)
        output_image_path = os.path.join(target_path, file_name)

        try:
            img = Image.open(input_image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {file_name}: {e}")
            continue

        if os.path.exists(output_image_path) and img.size == resolution:
            print(f"Skipped Iteration x{idx}. [Same Res]")
            continue

        resized_image = img.resize(resolution, Image.LANCZOS)  # type: ignore
        resized_image.save(output_image_path)

    print(f"Conversion Complete for {'Label: ' + label if is_labeled else 'Unlabeled'}")



def lower_all_images():
    for folderPath in os.listdir(Constants.DATA_PATH):
        lower_image_resolution(iterations=-1, resolution=Constants.RESOLUTION, images_path=os.path.join(Constants.DATA_PATH,folderPath),label=folderPath,output_path=Constants.OUTPUT_PATH)


def load_images_from_path(images_path: str, labeled: bool = False, single: bool = False):
    output = []
    labels = []

    if labeled:
        label_dirs = os.listdir(images_path)
        for label in label_dirs:
            label_path = os.path.join(images_path, label)
            for image_name in os.listdir(label_path):
                img = Image.open(os.path.join(label_path, image_name)).convert('RGB')
                arr = np.array(img)
                output.append(arr)
                labels.append(label)
        return np.array(output), np.array(labels)
    else:
        if single:
            img = Image.open(images_path).convert('RGB')
            arr = np.array(img)
            return np.array(output)
            
        for image_name in os.listdir(images_path):
            img = Image.open(os.path.join(images_path, image_name)).convert('RGB')
            arr = np.array(img)
            output.append(arr)
        return np.array(output)


def image_to_num(images_path: str):
    if len(os.listdir(images_path)) == 0:
        lower_all_images()

    return load_images_from_path(images_path, labeled=True)

def create_final_data():
    data,labels = image_to_num(images_path=Constants.OUTPUT_PATH)
    np.savez("pokemon_data.npz", images=data, labels=labels)

    print("Data file successfully created.")


def load_data():
    if not os.path.exists("pokemon_data.npz"):
        create_final_data()
    loaded = np.load("pokemon_data.npz")
    images = loaded['images']
    labels = loaded['labels']
    return images,labels
    
    