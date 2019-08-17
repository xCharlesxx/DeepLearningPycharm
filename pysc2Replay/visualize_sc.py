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
from keras.models import load_model
from random import randint

features = "screen"

autoencoder = load_model('model/autoencoder_' + features + '_32_128.h5')

feature_layers = 12
if features == "minimap":
    feature_layers = 8
width = 60
height = 60

x_train = []
x_test = []

data_files = glob.glob("data_test/*")
np.random.shuffle(data_files)
n = len(data_files)
i = 0
for data_file in data_files[:1]:
    game = pickle.load(open(data_file, "rb"))
    states = game["state"]
    f = [state[features] for state in states]
    x_test = x_test + f
    i += 1

x_train = np.array(x_train).reshape((len(x_train), feature_layers, width, height)).astype('float32')
x_test = np.array(x_test).reshape((len(x_test), feature_layers, width, height)).astype('float32')

decoded_imgs = autoencoder.predict(x_test)
x = randint(0, len(x_test))
decoded_img = decoded_imgs[x]
img = x_test[x]

n = feature_layers
plt.figure(figsize=(16, 6))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(img[i].reshape(width, height))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_img[i].reshape(width, height))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
