import numpy as np
from PIL import Image

import Constants
import convolutional_neural_network as cnn


model = cnn.ConvolutionNeuralNetwork(num_epochs=50)

def predict_from_image_path(image_path):
    img = Image.open(image_path).convert("RGB").resize((32, 32))
    arr = np.array(img)
    label = model.predict_label(arr)
    print(f"Predicted label: {label}")

predict_from_image_path(Constants.TEST_FILES_PATH + "\\unknown_pokemon.jpg")