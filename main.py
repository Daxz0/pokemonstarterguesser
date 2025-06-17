#libraries
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

## CONFIG
INPUT_PATH = "input/"
OUTPUT_PATH = "test_output/"

print(os.listdir(INPUT_PATH))

def lower_image_resolution(images_path=INPUT_PATH):
    for fileName in os.listdir(images_path):
        print(fileName)
        img = Image.open(images_path+fileName)
        resized_image = img.resize((224,224), Image.LANCZOS)
        resized_image.save(OUTPUT_PATH)
        
lower_image_resolution()

# cnn = CNNClassifier(num_epochs, layers, dropout)



# def load_data():
#   !wget -q --show-progress -O cifar_data https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%204%20_%205%20-%20Neural%20Networks%20_%20CNN/dogs_v_roads

#   import pickle
#   data_dict = pickle.load(open( "cifar_data", "rb" ));

#   data   = data_dict['data']
#   labels = data_dict['labels']

#   return data, labels

# data, labels = load_data()