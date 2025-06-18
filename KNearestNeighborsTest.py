from KNearestNeighbors import KNearestNeighbors
import numpy as np
import init
import Constants

x_data, y_data = init.load_data()

model = KNearestNeighbors(x_data, y_data)

best_score, best_model = model.find_highest_accuracy_score()
print("Best accuracy:", best_score)

neighbors_used = best_model.n_neighbors # type: ignore
accuracy, trained_model = model.k_nearest_neighbors_algorithm(neighbors_used)

model.save_model(trained_model)

model.generate_confusion_matrix()
