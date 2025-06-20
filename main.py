#libraries
import Constants
import init
import pandas as pd
import random
import os
import time
from collections import Counter
from PIL import Image

import numpy as np
import pandas as pd
import init
import joblib

import models.CNN.ConvolutionalNeuralNetwork as CNN


model = CNN.ConvolutionNeuralNetwork(num_epochs=50)

def predict_from_image_path(image_path):
    img = Image.open(image_path).convert("RGB").resize((32, 32))
    arr = np.array(img)
    label = model.predict_label(arr)
    print(f"Predicted label: {label}")



predict_from_image_path("unknown_pokemon.jpg")