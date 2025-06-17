import pandas
import Constants
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class KNearestNeighbors:
    def __init__(self, filename, x_label, y_label, test_size=Constants.TEST_SIZE, random_state=Constants.RANDOM_STATE):
        self.filename = filename
        self.x_label = x_label
        self.y_label = y_label
        self.test_size = test_size
        self.random_state = random_state
        self.x
        self.y

    def load_file(self):
        data_filename = self.filename
        data = pandas.read_csv(data_filename)
        self.x = data[[self.x_label]]
        self.y = data[[self.y_label]]

    def k_nearest_neighbors_algorithm(self, neighbors, x, y):
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, self.test_size, self.random_state)

        knn_model = KNeighborsClassifier(n_neighbors=neighbors)

        knn_model.fit(X_train, y_train)

        y_pred = knn_model.predict(X_test)

        return accuracy_score(y_test, y_pred)

    def find_highest_accuracy_score(self, max_neighbors=Constants.MAX_NEAREST_NEIGHBORS):
        max_accuracy_score = 0
        best_neighbors_value = 0

        for neighbors in range(1, max_neighbors, 2):
            accuracy_score = self.k_nearest_neighbors_algorithm(neighbors, self.x, self.y, self.test_size, self.random_state)
            if (accuracy_score > max_accuracy_score):
                max_accuracy_score = accuracy_score
                best_neighbors_value = neighbors

        return max_accuracy_score, best_neighbors_value
