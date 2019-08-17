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
from keras.callbacks import History
from keras import losses

history = History()

epochs = 10

features = "screen"
feature_layers = 12
if features == "minimap":
    feature_layers = 8
width = 60
height = 60

compression = 32
filters = 128

input = Input(shape=(feature_layers, width, height))

x = Conv2D(filters, (3, 3), activation='relu', padding='same', name="enc_1")(input)
x = MaxPooling2D((2, 2), padding='same', name="enc_2")(x)
x = Conv2D((int)(filters/2), (3, 3), activation='relu', padding='same', name="enc_3")(x)
x = MaxPooling2D((2, 2), padding='same', name="enc_4")(x)
x = Conv2D(compression, (3, 3), activation='relu', padding='same', name="enc_5")(x)
encoded = MaxPooling2D((2, 2), padding='same', name="encoded")(x)

print("Encoding shape: " + str(encoded.shape))

x = Conv2D(compression, (3, 3), activation='relu', padding='same', name="dec_1")(encoded)
x = UpSampling2D((2, 2), name="dec_2")(x)
x = Conv2D((int)(filters/2), (3, 3), activation='relu', padding='same', name="dec_3")(x)
x = UpSampling2D((2, 2), name="dec_4")(x)
x = Conv2D(filters, (3, 3), activation='relu', name="dec_5")(x)
x = UpSampling2D((2, 2), name="dec_6")(x)
decoded = Conv2D(feature_layers, (4, 4), activation='sigmoid', padding='same', name="decoded")(x)

autoencoder = Model(input, decoded)
autoencoder.compile(optimizer='adadelta', loss=losses.mean_squared_error)

# Load data
x_train = []
x_test = []

data_files = glob.glob("data/*")

n = (int)(len(data_files)/2)
i = 0
for data_file in data_files[:n]:
    game = pickle.load(open(data_file, "rb"))
    states = game["state"]
    f = [state[features] for state in states]
    if i > n * 0.75:
        x_test = x_test + f
    else:
        x_train = x_train + f
    i += 1

x_train = np.array(x_train).reshape((len(x_train), feature_layers, width, height)).astype('float32')
x_test = np.array(x_test).reshape((len(x_test), feature_layers, width, height)).astype('float32')

autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[history])

name = 'autoencoder_' + features + '_' + str(epochs) + '_' + str(compression) + '_' + str(filters)

autoencoder.save('model/' + name + '.h5')

print(history.history)
pickle.dump(history.history, open('log/' + name + '.p', 'wb'))

print("Model saved")