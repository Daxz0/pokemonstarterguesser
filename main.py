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

from KNearestNeighbors import KNearestNeighbors
import init

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



def save_knn_model():


    x_data, y_data = init.load_data()

    model = KNearestNeighbors(x_data, y_data)

    best_score, best_model = model.find_highest_accuracy_score()
    print("Best accuracy:", best_score)

    neighbors_used = best_model.n_neighbors # type: ignore
    accuracy, trained_model = model.k_nearest_neighbors_algorithm(neighbors_used)

    model.save_model(trained_model)

    model.generate_confusion_matrix()


images,labels = init.load_data()