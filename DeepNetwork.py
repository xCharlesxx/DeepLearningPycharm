from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers

import pickle

import keras as ks
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, Reshape, Concatenate, Input
from keras.callbacks import TensorBoard
from keras.layers.advanced_activations import LeakyReLU
from keras.backend.tensorflow_backend import set_session
from keras.constraints import nonneg
import gc
#from keras_transformer import get_model

from pysc2.lib import features
import numpy as np 
import os 
import csv
import natsort
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
        print("{}/{}".format(counter+1, all_files_size), end='\r')
        counter+=1
        full_path = os.path.join(training_data_dir, file)
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
                count += 1
            inputs.append(inputrows)

    print("{}/{}".format(counter, all_files_size))
    inputs = np.expand_dims(inputs, axis=3)

    inputs = np.reshape(inputs, (-1, const.InputSize(), const.InputSize(), 1))
    outputs = np.reshape(outputs, (-1, const.OutputSize()))

    inputs = inputs.astype(np.int)
    outputs = outputs.astype(np.float)

    return [inputs, outputs]

def get_training_data_dirs(training_data_dir):
    all_folders = os.listdir(training_data_dir)
    folders = []
    for folder in all_folders:
        files = []
        for file in natsort.natsorted(os.listdir(os.path.join(training_data_dir, folder))):
            files.append(os.path.join(training_data_dir, folder, file))
        folders.append(files)
    # all_files_size = len([num for num in all_files])
    return folders


def extract_data_dirs(dirs, num):
    # inputs = []
    # outputs = []
    inouts = [[], [[], []]]
    print('Extracting files...')
    for file in range(0, num):
        print('\r', end='')
        print(file, end ='')
        #print("{}/{}".format(counter+1, all_files_size), end='')

        inarr = np.load(dirs[file], allow_pickle=True)
        # outputs.append(inarr['action'])
        # inputs.append(inarr['feature_layers'])
        action = inarr['action']
        inouts[1][0].append(action[:5])
        inouts[1][1].append(action[5:])
        inouts[0].append(inarr['feature_layers'])
    # inputs = np.array(inputs)
    # outputs = np.array(outputs)
    inouts[0] = np.array(inouts[0])
    inouts[1][0] = np.array(inouts[1][0])
    inouts[1][1] = np.array(inouts[1][1])
    #print("Wait")
        #print("{}/{}".format(counter, all_files_size))
    return inouts#[inputs, outputs]

def sort_data_dirs():
    for file in get_training_data_dirs("C:\\Users\\Charlie\\training_data\\All"):
        arr = np.load(file, allow_pickle=True)
        b4 = arr['action']
        if len(b4) == 13:
            if arr['action'][1] == 1:
                arr.close()
                os.remove(file)
                print("Removed Box")
                continue
            if arr['action'][4] == 1:
                arr.close()
                os.remove(file)
                print("Removed HoldPos")
                continue
            afr = np.array([arr['action'][0], arr['action'][2], arr['action'][3],
                                      arr['action'][5], arr['action'][6], arr['action'][7],
                                      arr['action'][8], arr['action'][9], arr['action'][12]])
            np.savez_compressed(file, action=afr, feature_layers=arr['feature_layers'])


def translate_outputs_to_NN(output):
    trans_outputs = np.zeros([9], float)
    position = 0;
    for cell in output:
        try:
            for xy in cell:
                translated = position_translation(position, xy)
                if (translated[1] > 1):
                    translated[1] = 1
                    print("Something went wrong in translate_outputs")
                if(translated[1] < 0):
                    translated[1] = 0
                    print("Something went wrong in translate_outputs")
                trans_outputs[translated[0]] = translated[1]
                position += 1
                #print("{} -> {}".format(xy, translated))
        except TypeError:
            translated = position_translation(position, cell)
            if (translated[1] > 1 or translated[1] < 0):
                print("Something went wrong in translate_outputs")
            else:
                trans_outputs[translated[0]] = translated[1]
                position += 1
                #print("{} -> {}".format(cell, translated))
    if (output[0] in ability_dict):
        trans_outputs[8] = ability_dict[output[0]]
    return trans_outputs
def position_translation(position, value):
    #Action
    if (position == 0):
        #Select point
        if (value in [2]):
            return [0, 1]
        #Select rect
        # elif (value in [3]):
        #     return [1, 1]
        #Smart screen
        elif (value in [451, 452]):
            return [1, 1]
        #Attack screen
        elif (value in [13, 17, 12, 14, 16]):
            return [2, 1]
        #Hold pos
        # elif (value in [274,453]):
        #     return [4, 1]
        #Select army
        elif (value in [7]):
            return [3, 1]
        #Use ability
        else:
            return [4, 1]
    #Stack action
    if (position == 1):
        return [5, value[0] / 4]
    #Coordinates
    if (position >= 2):
        return [position + 4, value / const.ScreenSize().x]
    print("Something went wrong in position_translation")
    return 0

def get_closest_ability(ability_val, available_abilities):
    # Find available abilities to use
    abilities = list(set(available_abilities).intersection(ability_dict.keys()))
    ability_choice = 0
    best_dist = 1.0
    # Find ability with value closest to NN output value
    for ability in abilities:
        difference = abs(ability_dict[ability] - ability_val)
        if difference < best_dist:
            best_dist = difference
            ability_choice = ability
    if ability_choice in [45, 46, 47]:
        ability_choice = 45
    # Return ability chosen
    return ability_choice

ability_dict = {
    #inject
    204: 0,
    #tumour
    45: 0.5,
    46: 0.5,
    47: 0.5,
    # transfuse
    242: 1,
    # blinding cloud
    179: 0,
    #Abduct
    176: 0.3,
    #parasidic bomb
    215: 0.6,
    # viper consume
    243: 0.1,
    #Caustic spray
    184: 1,
    # corrosive bile
    188: 0.5,
    #explode
    191: 0.5,
    #fungal growth
    194: 0,
    #infested terrans
    203: 0.5,
    #neural parasite
    212: 1,
    #changeling
    215: 0,
    # contaminate
    188: 1,
    #locusts
    229: 0.5,
    #burrow down
    103: 0.9,
    #burrow up
    117: 1.0,
}
# select point
# 2: self.select_point,
# # select rect
# 3: self.double_select_point,
# # smart minimap
# 452: self.single_select_point,
# # attack minimap
# 13: self.single_select_point,
# 17: self.single_select_point,
# # smart screen
# 451: self.single_select_point,
# # attack screen
# 12: self.single_select_point,
# 14: self.single_select_point,
# 16: self.single_select_point,
# # Creep tumour
# 45: self.single_select_point,
# 46: self.single_select_point,
# 47: self.single_select_point,
# # Inject larve
# 204: self.single_select_point,
# # Burrow down
# 103: self.single_q,
# # Burrow up
# 117: self.single_q,
# # Blinding cloud
# 179: self.single_select_point,
# # Caustic spray
# 184: self.single_select_point,
# # Contaminate
# 188: self.single_select_point,
# # Corrosive bile
# 189: self.single_select_point,
# # Explode
# 191: self.single_q,
# # fungal growth
# 194: self.single_select_point,
# # infested terrans
# 203: self.single_select_point,
# # transfuse
# 242: self.single_select_point,
# # neural parasite
# 212: self.single_select_point,
# # parasidic bomb
# 215: self.single_select_point,
# # spawn changeling
# 228: self.single_q,
# # spawn locusts
# 229: self.single_select_point,
# # viper consume
# 243: self.single_select_point,
# # hold position quick
# 274: self.single_q,
# # stop quick
# 453: self.single_q,
# # select army
# 7: self.single_q
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
    padding = 'valid'
    activation = 'relu'
    data_format = 'channels_last'
    ks.constraints.non_neg()
    inputs = Input(shape=(352, 352, 12))

    convlayer1 = Conv2D(64, kernel_size=(5, 5))(inputs)
    convlayer1 = (LeakyReLU(alpha=0.3))(convlayer1)
    poollayer1 = (MaxPooling2D(pool_size=(2, 2), strides=None, padding=padding, data_format=data_format))(convlayer1)
    layer1 = (Dropout(0.3))(poollayer1)

    convlayer2 = (Conv2D(128, kernel_size=(3, 3)))(layer1)
    #merge = Concatenate()([layer1, convlayer2])
    convlayer2 = (LeakyReLU(alpha=0.3))(convlayer2)
    poollayer2 = (MaxPooling2D(pool_size=(2, 2), strides=None, padding=padding, data_format=data_format))(convlayer2)
    layer2 = (Dropout(0.3))(poollayer2)

    convlayer3 = (Conv2D(256, kernel_size=(3, 3)))(layer2)
    convlayer3 = (LeakyReLU(alpha=0.3))(convlayer3)
    poollayer3 = (MaxPooling2D(pool_size=(2, 2), strides=None, padding=padding, data_format=data_format))(convlayer3)
    layer3 = (Dropout(0.3))(poollayer3)

    layer4 = (Flatten())(layer3)
    layer4 = (Dense(256))(layer4)
    layer4 = (LeakyReLU(alpha=0.3))(layer4)

    layer5 = (Reshape((1, 256)))(layer4)
    # Add some memory
    layer5 = (LSTM(256))(layer5)

    output1 = (Dense(5, activation='softmax', name='Action'))(layer5)
    output2 = (Dense(4, activation='sigmoid', name='Parameters'))(layer5)
    #output2 = (LeakyReLU(alpha=0.3, name='Parameters'))(output2)

    model = Model(inputs, [output1, output2])
    model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                  optimizer="adam",
                  metrics=["accuracy"])
    model.summary()
    model.save("C:\\Users\\Charlie\\Models\\Conv2D-noobs")
    return model

def check_replay():
    TDDs = get_training_data_dirs("C:\\Users\\Charlie\\training_data\\4101\\")
    number = 0
    for TDD in TDDs:
        print("Replay: {}".format(number))
        number = number + 1
        # Get first frame of game
        TD = extract_data_dirs(TDDs[0], 1)
        for x in range(0, len(TD[0][0])):
            for y in TD[0][0][x]:
                # 5 = enemy units
                if y[5] == 1:
                    print("Corrupted")



def train_LSTM(loops = 100, loadSize = 4000, batch_size = 20, epochs = 1):
    # Dynamically grow the memory used on GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)
    model = ks.models.load_model("C:\\Users\\Charlie\\Models\\Conv2D-noobs - 1st Gen")
    for i in range(0, loops):
        TDDs = get_training_data_dirs("C:\\Users\\Charlie\\training_data\\4101\\")
        #random.shuffle(TDDs)
        # Whilst there's still data to train on
        while (len(TDDs) > 0):
            print("{} replays left".format(len(TDDs)))
            if len(TDDs[0]) < loadSize:
                TD = extract_data_dirs(TDDs[0], len(TDDs[0]))
                del TDDs[0]
                if TD[0].ndim == 4:
                    model.fit(TD[0], TD[1],
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_split=0.0,
                              shuffle=False, verbose=1)
                    print("Saved...")
                    model.save("C:\\Users\\Charlie\\Models\\Conv2D-noobs")
                TD.clear()
                gc.collect()

                # with open('/trainHistoryDict', 'wb') as file_pi:
                #     pickle.dump(history.history, file_pi)
            else:
                TD = extract_data_dirs(TDDs[0], loadSize)
                TDDs = TDDs[loadSize:]
                #del TDDs[0]
                model.fit(TD[0], TD[1],
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_split=0.0,
                          shuffle=False, verbose=1)
                TD.clear()
                gc.collect()
                # with open('/trainHistoryDict', 'wb') as file_pi:
                #     pickle.dump(history.history, file_pi)

    # del model
    # del TD
    # del TDDs
    # for i in range(20):
    #     gc.collect()
    return 0

#Tensorflow
# def build_net(input, info, num_action):
#     mconv1 = layers.conv2d(tf.transpose(input, [0, 2, 3, 1]),
#                             num_outputs=16,
#                             kernel_size=8,
#                             stride=4,
#                             scope='mconv1')
#     mconv2 = layers.conv2d(mconv1,
#                             num_outputs=32,
#                             kernel_size=4,
#                             stride=2,
#                             scope='mconv2')
#     info_fc = layers.fully_connected(layers.flatten(info),
#                                     num_outputs=256,
#                                     activation_fn=tf.tanh,
#                                     scope='info_fc')
#
#     # Compute spatial actions, non spatial actions and value
#     feat_fc = tf.concat([layers.flatten(mconv2), info_fc], axis=1)
#     feat_fc = layers.fully_connected(feat_fc,
#                                     num_outputs=256,
#                                     activation_fn=tf.nn.relu,
#                                     scope='feat_fc')
#
#     return spatial_action, non_spatial_action, value


# def build_transformer():
#
#     #TransformerBlock is a pseudo-layer combining together all nuts and bolts to assemble
#     #a complete section of both the Transformer and the Universal Transformer
#     #models, following description from the "Universal Transformers" paper.
#     #Each such block is, essentially:
#     #- Multi-head self-attention (masked or unmasked, with attention dropout,
#     #  but without input dropout)
#     #- Residual connection,
#     #- Dropout
#     #- Layer normalization
#     #- Transition function
#     #- Residual connection
#     #- Dropout
#     #- Layer normalization
#
#     transformer_block = TransformerBlock(
#         name='transformer',
#         num_heads=8,
#         residual_dropout=0.1,
#         attention_dropout=0.1,
#         use_masking=True)
#     add_coordinate_embedding = TransformerCoordinateEmbedding(
#         transformer_depth,
#         name='coordinate_embedding')
#
#     output = transformer_input # shape: (<batch size>, <sequence length>, <input size>)
#     for step in range(transformer_depth):
#         output = transformer_block(
#             add_coordinate_embedding(output, step=step))
#
#     return 0

