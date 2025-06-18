from KNearestNeighbors import KNearestNeighbors
import numpy as np
import init

x_data, y_data = init.load_data()

model = KNearestNeighbors(x_data, y_data)
print(model.find_highest_accuracy_score())
model.generate_confusion_matrix()
