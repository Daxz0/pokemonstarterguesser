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
import init
import joblib

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras # type: ignore
from keras.models import Sequential
from keras.layers import Dense, Conv2D, InputLayer
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


class ConvolutionNeuralNetwork:
    
    def __init__(self, num_epochs=30, layers=4, dropout=0.5):
        self.num_epochs = num_epochs
        self.layers = layers
        self.dropout = dropout
        self.label_encoder = LabelEncoder()
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Reshape((32, 32, 3)))

        for _ in range(self.layers):
            model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
            model.add(Activation('relu'))
        
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout))
        
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3 , 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.dropout))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(len(np.unique(Constants.LABELS_COUNT))))  # Output layer
        model.add(Activation('softmax'))

        opt = keras.optimizers.RMSprop(learning_rate=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model

    def preprocess_data(self, y_train, y_test):
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        y_train_onehot = to_categorical(y_train_encoded)
        y_test_onehot = to_categorical(y_test_encoded)
        return y_train_encoded, y_test_encoded, y_train_onehot, y_test_onehot

    def fit(self, X, y):
        
        self.preprocess_data()
        return self.model.fit(X, y, epochs=self.num_epochs, batch_size=10, verbose=2)
    
    def test(self, X):
        return self.model.predict(X)
    
    def score(self, X, y_true_encoded):
        y_pred_probs = self.test(X)
        y_pred_labels = np.argmax(y_pred_probs, axis=1)
        return accuracy_score(y_true_encoded, y_pred_labels)
