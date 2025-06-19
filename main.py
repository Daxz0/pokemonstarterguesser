#libraries
import Constants
import init
import pandas as pd
import random
import os
import time
from collections import Counter
from PIL import Image

import numpy as np
import pandas as pd

from KNearestNeighbors import KNearestNeighbors
import init
import joblib


# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import model_selection
# from sklearn.metrics import accuracy_score

# import tensorflow.keras as keras
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, InputLayer
# from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
# from keras.utils import to_categorical

def input_handler(knn_model):
    # converted_path = os.path.join(Constants.INPUT_PATH,"converted")
    converted_path = Constants.INPUT_PATH
    init.lower_image_resolution(
        iterations=-1,
        resolution=Constants.RESOLUTION,
        images_path=Constants.INPUT_PATH,
        output_path=converted_path,
        label=None
    )

    data = init.load_images_from_path(converted_path, labeled=False)
    data = np.reshape(data, (len(data), -1))

    predictions = knn_model.predict(data)
    print("Predictions:", predictions)
    return predictions


def save_knn_model():

    x_data, y_data = init.load_data()

    model = KNearestNeighbors(x_data, y_data)
    

    best_score, best_model = model.find_highest_accuracy_score()

    neighbors_used = best_model.n_neighbors # type: ignore
    score,trained_model = model.k_nearest_neighbors_algorithm(neighbors_used)

    model.save_model(trained_model)

    
def load_knn_model(filename="KNN_MODEL.pkl"):
    path = os.path.join(Constants.TRAINED_MODELS_OUTPUT, filename)
    if os.path.exists(path):
        print(f"Loading trained model from {path}")
        return joblib.load(path)
    else:
        print("Model not found. Training and saving a new one.")
        save_knn_model()
        return joblib.load(path)



# images,labels = init.load_data()
model = load_knn_model()
if model:
    input_handler(model)