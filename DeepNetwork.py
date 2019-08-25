from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers

import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, Reshape
from keras.callbacks import TensorBoard
#from keras_transformer import get_model

from pysc2.lib import features
import numpy as np 
import os 
import csv
import ast
import re
import random
from decimal import Decimal
from Constants import const
from ast import literal_eval

def get_training_data(training_data_dir):
    all_files = os.listdir(training_data_dir)
    all_files_size = len([num for num in all_files])
    inputs=[]
    outputs=[]
    counter = 0
    print('Extracting files...')
    for file in all_files:
        print("{}/{}".format(counter+1,all_files_size), end='\r')
        counter+=1
        full_path = os.path.join(training_data_dir,file)
        # Extract file code
        with open (full_path) as csv_file:
            reader = csv.reader(csv_file)
            count = 0
            inputrows =[]
            for row in reader:
                if count == const.InputSize():
                    outputs.append(row)
                else:
                    inputrows.append(row)
                count+=1
            inputs.append(inputrows)

    print("{}/{}".format(counter,all_files_size))
    inputs = np.expand_dims(inputs, axis=3)

    inputs = np.reshape(inputs, (-1,const.InputSize(),const.InputSize(),1))
    outputs = np.reshape(outputs, (-1,const.OutputSize()))

    inputs = inputs.astype(np.int)
    outputs = outputs.astype(np.float)

    return [inputs,outputs]

def get_training_data_layers(training_data_dir):
    all_files = os.listdir(training_data_dir)
    all_files_size = len([num for num in all_files])
    inputs = []
    outputs = []
    counter = 0
    print('Extracting files...')
    for file in all_files:
        print("{}/{}".format(counter+1, all_files_size), end='\r')
        counter += 1
        features = []
        action = []
        full_path = os.path.join(training_data_dir, file)
        inarr = np.load(full_path, allow_pickle=True)
        outputs.append(inarr[0][0])
        inputs.append(inarr[1:][0])
        # Extract file code
        # with open (full_path) as csv_file:
        #     reader = csv.reader(csv_file)
        #     layer = 0
        #     feature = []
        #     for index, row in enumerate(reader):
        #         if (index == 0):
        #             action = row
        #             action[0] = int(action[0])
        #             action[1] = int(action[1])
        #             action[2] = literal_eval(action[2])
        #             continue
        #         if ((index-1) % const.ScreenSize().y == 0 and index-1 != 0):
        #             layer += 1
        #             features.append(feature)
        #             feature = []
        #             continue
        #         feature.append([float(i) for i in row])
        #     features.append(feature)
        # inputs.append(features)
        # outputs.append(action)

    print("{}/{}".format(counter, all_files_size))
    # inputs = np.expand_dims(feature, axis=3)
    #
    # inputs = np.reshape(feature, (-1, const.InputSize(), const.InputSize(), 1))
    # outputs = np.reshape(action, (-1, const.OutputSize()))
    #
    # inputs = inputs.astype(np.int)
    # outputs = outputs.astype(np.float)

    return [inputs, outputs]

#Keras
#Shape = Shape of input data
#Dropout = Fraction rate of input inits to 0 at each update during training time, which prevents overfitting (0-1)
def build_knet():
    #TD = get_training_data_layers("training_data/EdgeCaseTest")
    dropout = 0.2
    learning_rate = 1e-4
    decay = 1e-6
    padding = 'same'
    loss_function = 'mean_squared_error'
    metrics = 'accuracy'
    epochs = 10
    batch_size = 2
    verbose = 1
    #Percent of data to be split for validation
    validation = 0.1
    
    training_data_dir = "training_data"
    tensorboard = TensorBoard(log_dir='logs/stage1')
    activation = 'tanh'

    model = Sequential()
    #model.add(Conv2D(32, (3, 3), padding=padding,
    #                 input_shape=(const.InputSize(), const.InputSize(), 1),
    #                 activation=activation))
    #model.add(Conv2D(32, (3, 3), activation=activation))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    ##model.add(Dropout(dropout))

    #model.add(Conv2D(64, (3, 3), padding=padding,
    #                 activation=activation))
    #model.add(Conv2D(64, (3, 3), activation=activation))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    ##model.add(Dropout(dropout))

    #model.add(Conv2D(128, (3, 3), padding=padding,
    #                 activation=activation))
    #model.add(Conv2D(128, (3, 3), activation=activation))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(dropout))


    #model.add(Dense(1280, activation=activation))

    #No defined input shape still works?! 
    model.add(Flatten())
    model.add(Dense(512, activation=activation))
   # model.add(Dense(320, activation=activation))
    model.add(Dense(160, activation=activation))
   # model.add(Dense(80, activation=activation))
    model.add(Dense(40, activation=activation))
   # model.add(Dense(20, activation=activation))
    model.add(Dense(10, activation=activation))
  #  model.add(Dense(5, activation=activation))
    #model.add(Dropout(dropout + 0.3))

    #Output Layer
    model.add(Dense(2, activation=activation))

    
    opt = ks.optimizers.adam(lr=learning_rate, decay=decay)

    model.compile(loss=loss_function, 
                  optimizer=opt, 
                  metrics=[metrics])

    file_size = len([num for num in os.listdir(training_data_dir)])
    TD = get_training_data(training_data_dir)
    #for sets in range(0,50):
        #print(inputs[sets])
        #print(outputs[sets])

    #xtrain = np.reshape(xtrain, (-1,const.InputSize(),const.InputSize(),1))

    model.fit(TD[0], TD[1], 
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.1, 
                shuffle=False, verbose=verbose)
    

    #prediction = model.predict(TD[0])

    #for loop in range(0,150):
    #    dec1 = np.around(prediction[loop][0],2)
    #    dec2 = np.around(prediction[loop][1],2)
    #    print(str([dec1,dec2]) + '      ' + str(TD[1][loop]))

    model.save("{}-{}-epochs-{}-batches-{}-dataSetSize".format(loss_function, epochs, batch_size, file_size))
    print('Finished Training')
    return 0

#multi-class classification problem
def build_LSTM():
    padding = 'same'
    activation = 'relu'
    model = Sequential()
    input = [features.Dimensions.screen,
             features.Dimensions.screen,
             features.Dimensions.screen,
             features.Dimensions.screen,
             features.Dimensions.screen,
             features.Dimensions.screen,
             features.Dimensions.screen,
             features.Dimensions.screen,]
    model.add(Conv2D(64, kernel_size=(5, 5), input_shape=(input),
                    activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2), padding=padding))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=(3, 3), 
                     activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2), padding=padding))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, kernel_size=(3, 3), activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2), padding=padding))
    model.add(Dropout(0.3))
    model.add(Flatten())

    model.add(Dense(256, activation=activation))
    model.add(Reshape((1, 256)))
    # Add some memory
    model.add(LSTM(256))
    model.add(Dense(2, activation=activation))
    model.compile(loss='mean_squared_error',
                  optimizer="adam",
                  metrics=["accuracy"])

    TD = get_training_data("raw_training_data")
    model.fit(TD[0], TD[1],
            batch_size=2,
            epochs=10,
            validation_split=0.1,
            shuffle=False, verbose=1)

    model.save("LSTM84")
    return model

#Tensorflow
def build_net(input, info, num_action):
    mconv1 = layers.conv2d(tf.transpose(input, [0, 2, 3, 1]),
                            num_outputs=16,
                            kernel_size=8,
                            stride=4,
                            scope='mconv1')
    mconv2 = layers.conv2d(mconv1,
                            num_outputs=32,
                            kernel_size=4,
                            stride=2,
                            scope='mconv2')
    info_fc = layers.fully_connected(layers.flatten(info),
                                    num_outputs=256,
                                    activation_fn=tf.tanh,
                                    scope='info_fc')

    # Compute spatial actions, non spatial actions and value
    feat_fc = tf.concat([layers.flatten(mconv2), info_fc], axis=1)
    feat_fc = layers.fully_connected(feat_fc,
                                    num_outputs=256,
                                    activation_fn=tf.nn.relu,
                                    scope='feat_fc')

    return spatial_action, non_spatial_action, value


def build_transformer():

    #TransformerBlock is a pseudo-layer combining together all nuts and bolts to assemble
    #a complete section of both the Transformer and the Universal Transformer
    #models, following description from the "Universal Transformers" paper.
    #Each such block is, essentially:
    #- Multi-head self-attention (masked or unmasked, with attention dropout,
    #  but without input dropout)
    #- Residual connection,
    #- Dropout
    #- Layer normalization
    #- Transition function
    #- Residual connection
    #- Dropout
    #- Layer normalization

    transformer_block = TransformerBlock(
        name='transformer',
        num_heads=8,
        residual_dropout=0.1,
        attention_dropout=0.1,
        use_masking=True)
    add_coordinate_embedding = TransformerCoordinateEmbedding(
        transformer_depth,
        name='coordinate_embedding')
    
    output = transformer_input # shape: (<batch size>, <sequence length>, <input size>)
    for step in range(transformer_depth):
        output = transformer_block(
            add_coordinate_embedding(output, step=step))

    return 0

