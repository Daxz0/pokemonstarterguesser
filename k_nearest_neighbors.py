import Constants
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import init
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbors:
    def __init__(self, file_path, X_label, y_label, test_size=Constants.TEST_SPLIT, random_state=Constants.RANDOM_STATE):
        self.data = np.load(file_path)
        self.X_data = self.data[X_label]
        self.y_data = self.data[y_label]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.test_size = test_size
        self.random_state = random_state
        self.model = None

    def return_accuracy_score(self):
        return accuracy_score(self.y_test, self.y_pred)

    def predict(self, image_path):
        image_data = init.image_encoder(images_path=image_path, single=True)
        # image_data = init.lower_image_resolution(iterations=-1, resolution=(32, 32), images_path=image_path, output_path=image_path, single=True)
        image_data = np.reshape(image_data, (len(image_data), -1))
        #image_data = image_data.reshape(len(image_data), len(image_data[0]), 3)
        #image_data = image_data
        return self.model.predict(image_data)

    def generate_confusion_matrix(self):
        if self.y_test is None or self.y_pred is None:
            raise RuntimeError("Run the algorithm before generating the confusion matrix.")

        labels = np.unique(self.y_test)
        cnf_matrix = confusion_matrix(self.y_test, self.y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=labels)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.show()

    def _shuffle_data(self):
        indices = np.random.permutation(len(self.X_data))
        self.X_data = self.X_data[indices]
        self.y_data = self.y_data[indices]

    def _build_model(self, neighbors):
        self._shuffle_data()
        self.model = KNeighborsClassifier(neighbors)
        self.X_data = np.reshape(self.X_data, (len(self.X_data), -1))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_data, self.y_data, test_size=self.test_size, random_state=self.random_state)

    def _fit(self):
        self.model.fit(self.X_train, self.y_train)

    def _test(self):
        self.y_pred = self.model.predict(self.X_test)
    
    def _model_machine_learning_pipeline(self, neighbors):
        self._build_model(neighbors)
        self._fit()
        self._test()

    def _find_most_optimal_solution(self, max_neighbors=Constants.MAX_NEAREST_NEIGHBORS):
        max_accuracy_score = 0
        most_optimal_neighbors = 0

        for neighbors in range(1, max_neighbors, 2):
            self._model_machine_learning_pipeline(neighbors)
            score = self.return_accuracy_score()

            if score > max_accuracy_score:
                max_accuracy_score = score
                most_optimal_neighbors = neighbors

        self._build_model(most_optimal_neighbors)
