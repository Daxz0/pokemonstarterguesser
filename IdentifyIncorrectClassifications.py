import KNearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def identify_incorrect_classifications():
    y_pred = KNearestNeighbors.get_y_pred()
    x_test = KNearestNeighbors.get_x_test()
    y_test = KNearestNeighbors.get_y_test()

    images = []
    incorrect_classifications = []
    actual_classifications = []

    for counter in range(len(x_test)):
        if y_pred[counter] != y_test[counter]:
            images.append(x_test[counter])
            incorrect_classifications.append(y_pred[counter])
            actual_classifications.append(y_test[counter])

    np.savez('incorrect_classifications.npz', images=images, incorrect_labeles=incorrect_classifications, actual_labels=actual_classifications)

    return 'incorrect_classifications.npz'

def display_incorrect_classifications(filename):
    for counter in range()