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


data, labels = init.load_data()
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=None)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

y_train_onehot = to_categorical(y_train_encoded)
y_test_onehot = to_categorical(y_test_encoded)


class ConvolutionNeuralNetwork:
    
    def __init__(self, num_epochs=30, layers=4, dropout=0.5):
        self.num_epochs = num_epochs
        self.layers = layers
        self.dropout = dropout
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Reshape((32, 32, 3)))

        for i in range(self.layers):
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
        model.add(Dense(len(np.unique(labels)))) #dense for every label we have
        model.add(Activation('softmax'))
        opt = keras.optimizers.RMSprop(learning_rate=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model
    
    def fit(self, *args, **kwargs):
        return self.model.fit(*args, epochs=self.num_epochs, batch_size=10, verbose=2, **kwargs)
    
    def test(self, *args, **kwargs):
        predictions = self.model.predict(*args, **kwargs)
        return predictions
    
    def score(self, X, y):
        predictions = self.test(X)
        return accuracy_score(y, predictions)
    


cnn = ConvolutionNeuralNetwork(num_epochs=50, layers=5)
cnn.fit(X_train, y_train_onehot)

y_pred_probs = cnn.test(X_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_test_encoded, y_pred_labels)
print(f"Test Accuracy: {accuracy:.2f}")
