import pandas
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class KNearestNeighbors:
    def __init__(self, filename, x_label, y_label):
        self.filename = filename
        self.x_label = x_label
        self.y_label = y_label
        self.x
        self.y

    def load_file(self):
        data_filename = self.filename
        data = pandas.read_csv(data_filename)
        self.x = data[[self.x_label]]
        self.y = data[[self.y_label]]

    def k_nearest_neighbors_algorithm(self, neighbors, x, y):
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

        knn_model = KNeighborsClassifier(n_neighbors=neighbors) # Experiment with different numbers of neighbors!

        knn_model.fit(X_train, y_train)

        y_pred = knn_model.predict(X_test)

        return accuracy_score(y_test, y_pred)

    def find_highest_accuracy_score(self):
        max_accuracy_score = 0
        for counter in range(1, 21):
            accuracy_score = self.k_nearest_neighbors_algorithm(counter, self.x, self.y)
            if (accuracy_score > max_accuracy_score):
                max_accuracy_score = accuracy_score

        return max_accuracy_score
