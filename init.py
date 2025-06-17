#libraries
import Constants
import pandas as pd
import os
from collections import Counter
from PIL import Image
import numpy as np
import pandas as pd


def lower_image_resolution(interations: int, resolution: tuple, images_path: str, label: str) -> None:
    all_paths = os.listdir(images_path)
    if interations < 0:
        interations = len(all_paths)
    for fileName in zip(range(interations),all_paths):
        img = Image.open(str(os.path.join(images_path,str(fileName[1]))))
        parent_path = os.path.join(Constants.OUTPUT_PATH,label)
        final_path = os.path.join(parent_path,str(fileName[1]))
        
        if os.path.exists(final_path):
            if img.size == resolution:
                print(f"Skipped Interation x{fileName[0]}. [Same Res]")
                continue
        
        
        os.makedirs(parent_path, exist_ok=True)
        
        resized_image = img.resize(resolution, Image.LANCZOS).convert('RGB')
        resized_image.save(final_path)
    print(f"Conversion Complete For Label: {label}")


def lower_all_images() -> bool:
    if len(os.listdir(Constants.OUTPUT_PATH)) > 0:
        return False
    for folderPath in os.listdir(Constants.INPUT_PATH):
        lower_image_resolution(interations=-1, resolution=(32,32), images_path=os.path.join(Constants.INPUT_PATH,folderPath),label=folderPath)
    return True


def image_to_num():
    images_list = os.listdir(Constants.OUTPUT_PATH)
    output = []
    labels = []

    for label in images_list:
        for image_name in os.listdir(os.path.join(Constants.OUTPUT_PATH, label)):
            img = Image.open(os.path.join(Constants.OUTPUT_PATH, label, image_name)).convert('RGB')
            arr = np.array(img)
            output.append(arr)
            labels.append(label)
    return np.array(output), np.array(labels)

def create_final_data():
    data,labels = image_to_num()
    np.savez("pokemon_data.npz", images=data, labels=labels)

    print("Data file successfully created.")

# loaded = np.load("pokemon_data.npz")
# images = loaded['images']
# labels = loaded['labels']