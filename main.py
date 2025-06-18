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

def input_handler(model: type):
    init.lower_image_resolution(iterations=-1, resolution=Constants.RESOLUTION,
                           images_path=Constants.INPUT_PATH, output_path=Constants.INPUT_PATH, label="Input")

    data = init.load_images_from_path(Constants.INPUT_PATH, labeled=False)

    model.find_highest_accuracy_score()
    return model.get_y_pred()


images,labels = init.load_data()