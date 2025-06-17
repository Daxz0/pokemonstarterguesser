#libraries
import Constants
import pandas as pd
import random
import os
import time
from collections import Counter
from PIL import Image

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score

import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, InputLayer
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from keras.utils import to_categorical



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
        for image in os.listdir(os.path.join(Constants.OUTPUT_PATH,label)):
            # print(os.path.join(Constants.OUTPUT_PATH,label,image))
            img = Image.open(os.path.join(Constants.OUTPUT_PATH,label,image)).convert('RGB')
            width, height = img.size
            rgb_values = [img.getpixel((x, y)) for y in range(height) for x in range(width)]
            output.append(rgb_values)
            labels.append(label)
    return output, labels

# lower_all_images()


image_array = image_to_num()
print(image_array)
    
            
    
