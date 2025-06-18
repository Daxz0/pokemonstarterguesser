import Constants
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbors:
    def __init__(self, x_data=None, y_data=None, file=None, x_label=None, y_label=None,
                 test_size=Constants.TEST_SPLIT, random_state=Constants.RANDOM_STATE):
        if file is not None and x_label is not None and y_label is not None:
            self.x = file[[x_label]].values
            self.y = file[[y_label]].values.ravel()
        elif x_data is not None and y_data is not None:
            self.x = x_data
            self.y = y_data
        else:
            raise ValueError("Insufficient input: Provide either file with labels or x_data and y_data.")

        self._init_standard_properties(test_size, random_state)

    def _init_standard_properties(self, test_size, random_state):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.test_size = test_size
        self.random_state = random_state
        self.model = None

    def get_X_test(self):
        return self.X_test

    def get_Y_test(self):
        return self.y_test

    def get_y_pred(self):
        return self.y_pred

    def shuffle_data(self):
        indices = np.random.permutation(len(self.x))
        self.x = self.x[indices]
        self.y = self.y[indices]

    def create_model(self, neighbors=Constants.MAX_NEAREST_NEIGHBORS):
        return KNeighborsClassifier(n_neighbors=neighbors)

    def save_model(self, model, filename="KNN_MODEL.pkl"):
        path = os.path.join(Constants.TRAINED_MODELS_OUTPUT,filename)
        if model is None:
            model = self.model
        if model is not None:
            joblib.dump(model, path)
        else:
            raise ValueError("No model to save.")

    def load_model(self, filename="KNN_MODEL.pkl", fallback_neighbors=Constants.MAX_NEAREST_NEIGHBORS):
        path = os.path.join(Constants.TRAINED_MODELS_OUTPUT,filename)
        if os.path.exists(path):
            return joblib.load(path)
        return self.create_model(neighbors=fallback_neighbors)

    def k_nearest_neighbors_algorithm(self, neighbors, x=None, y=None):
        self.shuffle_data()

        x = self.x if x is None else x
        y = self.y if y is None else y

        x = np.reshape(x, (len(x), -1))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state)

        self.model = self.load_model(fallback_neighbors=neighbors)
        self.model.fit(self.X_train, self.y_train)

        self.y_pred = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, self.y_pred), self.model

    def find_highest_accuracy_score(self, max_neighbors=Constants.MAX_NEAREST_NEIGHBORS):
        max_accuracy_score = 0
        best_model = None

        for neighbors in range(1, max_neighbors, 2):
            score, model = self.k_nearest_neighbors_algorithm(neighbors)
            if score > max_accuracy_score:
                max_accuracy_score = score
                best_model = model

        self.model = best_model
        return max_accuracy_score, best_model

    def generate_confusion_matrix(self):
        if self.y_test is None or self.y_pred is None:
            raise RuntimeError("Run the algorithm before generating the confusion matrix.")

        labels = np.unique(self.y_test)
        cnf_matrix = confusion_matrix(self.y_test, self.y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=labels)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.show()
