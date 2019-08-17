from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.backend import image_dim_ordering
import pickle
import glob
from keras.callbacks import History, EarlyStopping
import multiprocessing
import math
from keras.models import load_model

max_epochs = 50
batch_size = 32
chunk_size = 100

features = "screen"

autoencoder = load_model('model/autoencoder_' + features + '_32_128.h5')

feature_layers = 12
if features == "minimap":
    feature_layers = 8
width = 60
height = 60

# Raw data load
def load_data(folder):
    data_files = glob.glob(folder + "/*")
    i = 0
    data = []
    for data_file in data_files:
        game = pickle.load(open(data_file, "rb"))
        states = game["state"]
        f = [state[features] for state in states]
        data = data + f
    return np.array(data).reshape((len(data), feature_layers, width, height)).astype('float32')

# Test
test_data = load_data("data_test")
test_loss = autoencoder.evaluate(x=test_data, y=test_data, batch_size=batch_size, verbose=1)
print("Test loss = " + str(test_loss))