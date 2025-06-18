import pandas
import Constants
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

class KNearestNeighbors:
    def __init__(self, file, x_label, y_label, test_size=Constants.TEST_SPLIT, random_state=Constants.RANDOM_STATE):
        self.x = file[[x_label]]
        self.y = file[[y_label]]
        self._init_standard_properties(test_size, random_state)

    def __init__(self, x_data, y_data, test_size=Constants.TEST_SPLIT, random_state=Constants.RANDOM_STATE):
        self.x = x_data
        self.y = y_data
        self._init_standard_properties(test_size, random_state)

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

    def k_nearest_neighbors_algorithm(self, neighbors, x, y):
        self.shuffle_data()

        x_len = len(x)
        x = x.reshape(x_len, -1)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)

        knn_model = KNeighborsClassifier(n_neighbors=neighbors)

        knn_model.fit(self.X_train, self.y_train)

        self.y_pred = knn_model.predict(self.X_test)

        return accuracy_score(self.y_test, self.y_pred)

    def find_highest_accuracy_score(self, max_neighbors=Constants.MAX_NEAREST_NEIGHBORS):
        max_accuracy_score = 0

        for neighbors in range(1, max_neighbors, 2):
            accuracy_score = self.k_nearest_neighbors_algorithm(neighbors, self.x, self.y)
            if (accuracy_score > max_accuracy_score):
                max_accuracy_score = accuracy_score

        return max_accuracy_score

    def generate_confusion_matrix(self):
        labels = np.unique(self.y_test)

        cnf_matrix = confusion_matrix(self.y_test, self.y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=labels)
        disp.plot(cmap='Blues')
        print("Generating confusion matrix...")

        plt.show()

    def _init_standard_properties(self, test_size, random_state):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.test_size = test_size
        self.random_state = random_state