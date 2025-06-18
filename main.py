#libraries
import Constants
import pandas as pd
import random
import os
import time
from collections import Counter
from PIL import Image

import numpy as np
import pandas as pd

from init import load_data

# import seaborn as sns
# import matplotlib.pyplot as plt

# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import model_selection
# from sklearn.metrics import accuracy_score

# import tensorflow.keras as keras
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, InputLayer
# from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
# from keras.utils import to_categorical


images,labels = load_data()