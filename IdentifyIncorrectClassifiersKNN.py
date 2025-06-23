from KNearestNeighbors import *
import init
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

x_data, y_data = init.load_data()
# model = KNearestNeighbors(x_data, y_data)
# model.find_highest_accuracy_score()

def identify_incorrect_classifications(model: type):
    y_pred = model.get_y_pred()
    x_test = model.get_X_test()
    y_test = model.get_Y_test()

    images = []
    incorrect_classifications = []
    actual_classifications = []

    for counter in range(len(x_test)):
        if y_pred[counter] != y_test[counter]:
            images.append(x_test[counter])
            incorrect_classifications.append(y_pred[counter])
            actual_classifications.append(y_test[counter])

    np.savez('incorrect_classifications.npz', images=images, incorrect_labels=incorrect_classifications, actual_labels=actual_classifications)

    return 'incorrect_classifications.npz'

def display_incorrect_classifications(filename):
    data = np.load(filename)
    images = data['images']
    incorrect_labels = data['incorrect_labels']

    images = images.reshape(-1, 32, 32, 3)
    num_images = len(images)
    cols = 5
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    axes = axes.flatten()

    for i in range(data['images'].shape[0]):
        img = images[i].astype(np.uint8)
        axes[i].imshow(img)
        axes[i].axis('off')

        axes[i].text(
        0.5, -0.1,  # X=50% width, Y=just below the image
        str(incorrect_labels[i]),
        transform=axes[i].transAxes,
        ha='center',
        va='top',
        fontsize=10,
        color='black'
    )

    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.8, bottom=0.05)
    plt.savefig("all_images.png", dpi=150)
    plt.show()
