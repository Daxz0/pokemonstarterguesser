#libraries
import pandas as pd
import random
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

#set TF_ENABLE_ONEDNN_OPTS=0  # For Windows
#export TF_ENABLE_ONEDNN_OPTS=0  # For Linux/macOS


# cnn = CNNClassifier(num_epochs, layers, dropout)
print("test")


# def load_data():
#   !wget -q --show-progress -O cifar_data https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%204%20_%205%20-%20Neural%20Networks%20_%20CNN/dogs_v_roads

#   import pickle
#   data_dict = pickle.load(open( "cifar_data", "rb" ));

#   data   = data_dict['data']
#   labels = data_dict['labels']

#   return data, labels

# data, labels = load_data()

#Stuff for MLP training

#class MLPModel:
    #def __init__(self, hidden_layers=(100, 100, 100), max_iter=1000, test_size=0.2, random_state=1):
        #self.hidden_layers = hidden_layers
        #self.max_iter = max_iter
        #self.test_size = test_size
        #self.random_state = random_state
        #self.scaler = StandardScaler()
        #self.model = None

#X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)

#p_model = MLPClassifier(hidden_layer_sizes=(3, 4))

#for i in [(100, 100, 100), ]:
  #p_model = MLPClassifier(hidden_layer_sizes=(i), random_state=1, max_iter=1000000)
  #p_model.fit(X_train, y_train)
  #y_pred = p_model.predict(X_test)
  #return("Accuracy: ", accuracy_score(y_test, y_pred))

