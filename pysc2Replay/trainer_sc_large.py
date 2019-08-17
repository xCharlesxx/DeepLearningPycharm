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
from keras import losses

max_epochs = 50
batch_size = 32
chunk_size = 100

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

#autoencoder = load_model('model/autoencoder_' + features + '_32_128.h5')

def get_data_size(folder):
    data_files = glob.glob(folder + "/*")
    n = len(data_files)
    file = data_files[0]
    game = pickle.load(open(file, "rb"))
    states = game["state"]
    return len(data_files) * len(states)

# Generator for training data
def batchGenerator(folder):

    # Data files
    data_files = glob.glob(folder+"/*")
    n = len(data_files)
    chunks = (int)(math.ceil(n/chunk_size))

    while True:
        np.random.shuffle(data_files)
        for i in range(chunks):
            idx = i * chunk_size
            l = min(n - idx, chunk_size)
            chunk = data_files[idx:idx+l]
            data = []
            for file in chunk:
                game = pickle.load(open(file, "rb"))
                states = game["state"]
                f = [state[features] for state in states]
                data += f
            data = np.array(data).reshape((len(data), feature_layers, width, height)).astype('float32')
            np.random.shuffle(data)
            batches_in_chunk = (int)(math.ceil(len(data)/batch_size))
            for batch in range(batches_in_chunk):
                batch_start = batch*batch_size
                batch_end = min(batch_start+batch_size, len(data))
                batch_data = data[batch_start:batch_end]
                batch_data = np.array(batch_data).reshape((len(batch_data), feature_layers, width, height)).astype('float32')
                yield batch_data, batch_data

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

# Callbacks
history = History()
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

# Train
autoencoder.fit_generator(batchGenerator("../data_train"),
                epochs=max_epochs,
                steps_per_epoch=get_data_size("../data_train") / batch_size,
                validation_data=batchGenerator("../data_validation"),
                validation_steps=get_data_size("../data_validation") / batch_size,
                use_multiprocessing=True,
                workers=(int)(multiprocessing.cpu_count() / 2),
                callbacks=[history, early_stopping])

# Save stuff
name = 'autoencoder_' + features + '_' + str(compression) + '_' + str(filters)
autoencoder.save('model/' + name + '.h5')
print(history.history)
pickle.dump(history.history, open('log/' + name + '.p', 'wb'))
print("Model saved")

# Test
test_data = load_data("../data_test")
test_loss = autoencoder.evaluate(x=test_data, y=test_data, batch_size=batch_size)
print("Test loss = " + str(test_loss))