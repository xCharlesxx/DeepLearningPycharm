from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import pysc2
import numpy
#import cv2
#import pandas
from pprint import pprint

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

import csv
import random
from decimal import Decimal
from Constants import const
from absl import app

class MoveToBeacon(base_agent.BaseAgent):
    loaded = False
        #Pysc2 defs
    def get_obs(self, obs):
        return {self.screen: obs['screen'],
                self.available_actions: obs['available_actions']}

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.multi_select) > 0): 
            if (obs.observation.multi_select[0].unit_type == unit_type):
                return True
        elif (len(obs.observation.single_select) > 0):
                if (obs.observation.single_select[0].unit_type == unit_type):
                    return True
        
      #Tensorflow defs
    def build():
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        self.saver = tf.train.Saver(variables, max_to_keep=100)
        self.init_op = tf.variables_initializer(variables)
        train_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)
        self.train_summary_op = tf.summary.merge(train_summaries)

    def save(self, path, step=None):
        os.makedirs(path, exist_ok=True)
        step = step or self.train_step
        print("Save agent to %s, step %d" % (path, step))
        ckpt_path = os.path.join(path, 'model.ckpt')
        self.saver.save(self.sess, ckpt_path, global_step=step)

    def load(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        self.train_step = int(ckpt.model_checkpoint_path.split('-')[-1])
        print("Load agent at step %d" % self.train_step)

    def loadK(self, path):
        self.model = ks.models.load_model(path)

    def stencil(self, _stencil, _raw_list):
        input = _raw_list
        stencil = _stencil
        newInput = numpy.zeros((const.InputSize(),const.InputSize()),int)
        counterx = 0
        countery = 0
        for numy, y in enumerate(stencil):
            for numx, x in enumerate(y): 
                if (x != 0):
                    newInput[countery][counterx] = input[numy][numx]
                    counterx+=1
                if (counterx == const.InputSize()):
                    countery+=1
                    counterx=0
        return newInput

    def step(self, obs):
        super(MoveToBeacon, self).step(obs)

        if MoveToBeacon.loaded == False:
            self.loadK("8Dense-10-epochs-2-batches-3000-dataSetSize-98%")
            MoveToBeacon.loaded = True
        input = obs.observation.feature_minimap[5]
        for x in input:
                output = ""
                for i in x:
                    output+=str(i)
                    output+=""
                print(output)
        print("\n")        
        #If maring is selected, use DNN
        #print("Single: " + (str)(len(obs.observation.single_select)))
        #print("Multi: " + (str)(len(obs.observation.multi_select)))
        if self.unit_type_is_selected(obs, units.Terran.Marine):
            if self.can_do(obs, actions.FUNCTIONS.Attack_screen.id):

                newInput = self.stencil(obs.observation.feature_minimap[3], obs.observation.feature_minimap[5])

                #for x in stencil:
                #        output = ""
                #        for i in x:
                #            output+=str(i)
                #            output+=""
                #        print(output)
                #print("\n")
                #for x in input:
                #        output = ""
                #        for i in x:
                #            output+=str(i)
                #            output+=""
                #        print(output)
                #print("\n")
                #for x in newInput:
                #        output = ""
                #        for i in x:
                #            output+=str(i)
                #            output+=" "
                #        print(output)
                #print("\n")

                newInput = numpy.expand_dims(newInput, axis=2)
                newInput = newInput.reshape([-1,const.InputSize(),const.InputSize(),1])
                prediction = self.model.predict(newInput)
                outputx = prediction[0][0] * const.ScreenSize()
                outputy = prediction[0][1] * const.ScreenSize()
                #print(('Network Prediction vs Optimum: ({},{}) ({},{})'.format(int(outputx),int(outputy),beacon.x,beacon.y)), end='\r')
                return actions.FUNCTIONS.Attack_screen("now", (outputx,outputy))
        #Select Marine
        else: 
            marine = self.get_units_by_type(obs, units.Terran.Marine)
            if len(marine) > 0: 
                return actions.FUNCTIONS.select_point('select_all_type', (marine[0].x,
                                                                    marine[0].y))

        #features.MinimapFeatures.
        return actions.FUNCTIONS.no_op()

class GenerateMoveToBeaconTestData(base_agent.BaseAgent):
        #Pysc2 defs
    packagedInput = numpy.zeros((const.InputSize(),const.InputSize()),int)
    packagedOutput = numpy.empty(const.OutputSize(), float)
    packageCounter = 0
    def get_obs(self, obs):
        return {self.screen: obs['screen'],
                self.available_actions: obs['available_actions']}

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
         obs.observation.single_select[0].unit_type == unit_type):
          return True

    def step(self, obs):
        super(GenerateMoveToBeaconTestData, self).step(obs)


       # Extract file code
       # with open ('training_data/0.csv') as csv_file:
       #     reader = csv.reader(csv_file)
       #     count = 0
       #     for row in reader:
       #         if count == 0:
       #             GenerateMoveToBeaconTestData.packagedInput = row
       #         if count == 2:
       #             GenerateMoveToBeaconTestData.packagedOutput = row
       #         count+=1


        #If previous action was successful, record as training data
        if obs.reward > 0:
            #counter = 0
            #output = ""
            #for x in GenerateMoveToBeaconTestData.packagedInput:
            #    output+=str(x)
            #    output+=" "
            #    counter+=1 
            #    if (counter == const.InputSize()):
            #        print(output)
            #        output = ""
            #        counter = 0
            #print(GenerateMoveToBeaconTestData.packagedOutput)
            fileName = 'raw_training_data/' + str(self.packageCounter) + '.csv'
            with open(fileName, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.packagedInput)
                writer.writerow(self.packagedOutput)
            self.packageCounter+=1
            #if (self.packageCounter == 3000):
            #    self.packageCounter = 4000


        newinput = obs.observation.feature_screen[6]


        #stencil = obs.observation.feature_minimap[3]

        ##24x24 is refined input data size
        ##Use camera stencil to grab relevent data   
        #newinput = numpy.zeros((const.InputSize(),const.InputSize()),int)
        #counterx = 0
        #countery = 0
        #for numy, y in enumerate(stencil):
        #    for numx, x in enumerate(y): 
        #        if (x == 1):
        #            newinput[countery][counterx] = input[numy][numx]
        #            counterx+=1
        #        if (counterx == const.InputSize()):
        #            countery+=1
        #            counterx=0

        for unit in obs.observation.feature_units:
            if(unit.unit_type == 317):
                beacon = unit

        
        #for x in newinput:
        #    output = ""
        #    for i in x:
        #        output+=str(i)
        #        output+=""
        #    print(output)
        #print("\n")

        #Screen is not 84x84 but ~84x60 but 84x84 for simplicity
        outputx = beacon.x #random.randint(0,const.ScreenSize())
        outputy = beacon.y #random.randint(0,const.ScreenSize())

        self.packagedInput = newinput
        #/84 to get a number between 0 and 1 as outputs for DNN
        self.packagedOutput = [float(round(Decimal(outputx/const.ScreenSize()),2)),
                                                       float(round(Decimal(outputy/const.ScreenSize()),2))]
        if self.unit_type_is_selected(obs, units.Terran.Marine):
            if self.can_do(obs, actions.FUNCTIONS.Attack_screen.id):
                return actions.FUNCTIONS.Attack_screen("now", (outputx,outputy))
        #Select Marine
        else: 
            marine = self.get_units_by_type(obs, units.Terran.Marine)
            if len(marine) > 0: 
                return actions.FUNCTIONS.select_point('select_all_type', (marine[0].x,
                                                                    marine[0].y))


        #features.MinimapFeatures.
        return actions.FUNCTIONS.no_op()