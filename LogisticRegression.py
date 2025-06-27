import pandas
import Constants
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, filename, x_label, y_label, test_size=Constants.TEST_SPLIT, random_state=Constants.RANDOM_STATE):
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

    def logistic_regression_algorithm(self):
        
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, self.test_size, self.random_state)

        model = LogisticRegression()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        return accuracy_score(y_test, y_pred)
    
    def generate_confusion_matrix(self, y_true, y_pred):
        cnf_matrix = confusion_matrix(y_true, y_pred, labels = [0, 1, 2, 3, 4])
        disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=['0 (Pikachu)', '1 (Charmander)', '2 (Bulbasaur)', '3 (Squirtle)'])
        disp.plot (cmap='Blues')
        plt.show()