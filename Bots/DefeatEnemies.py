from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras as ks
import numpy as np

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

class DefeatEnemies(base_agent.BaseAgent):
    loaded = False
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
    def loadK(self, path):
        self.model = ks.models.load_model(path)

    def step(self, obs):
        super(DefeatEnemies, self).step(obs)

        if DefeatEnemies.loaded == False:
            self.loadK("LSTM84")
            DefeatEnemies.loaded = True

        if self.unit_type_is_selected(obs, units.Terran.Marine):
            if self.can_do(obs, actions.FUNCTIONS.Attack_screen.id):

                #84x84 Detailed
                #Things to be known:
                #Height_map(0) visibility(1) creep(2) Player_relative(5) Unit_type(6) selected(7) hit_points(8) unit_density(11) 
                input = obs.observation.feature_screen[6]
                #features.Dimensions.screen
                #31x31 Simplified
                #input = obs.observation.feature_screen[5]
                #stencil = obs.observation.feature_screen[3]
                #newInput = numpy.zeros((const.InputSize(),const.InputSize()),int)
                #counterx = 0
                #countery = 0
                #for numy, y in enumerate(stencil):
                #    for numx, x in enumerate(y): 
                #        if (x == 1):
                #            newInput[countery][counterx] = input[numy][numx]
                #            counterx+=1
                #        if (counterx == const.InputSize()):
                #            countery+=1
                #            counterx=0
                obs.observation
                for x in obs.observation.feature_screen[6]:
                        output = ""
                        for i in x:
                            output+=str(i)
                            output+=""
                        print(output)
                print("\n")

                newInput = numpy.expand_dims(input, axis=2)
                newInput = newInput.reshape([-1,const.InputSize(),const.InputSize(),1])


                #self.model.fit(TD[0], TD[1], 
                #batch_size=batch_size,
                #epochs=epochs,
                #validation_split=0.1, 
                #shuffle=False, verbose=verbose)
                prediction = self.model.predict(newInput)
                #for marines in obs.observation.multi_select:
                #    obs.observation.multi_select[0]
                #    prediction = self.model.predict(newInput)


                outputx = prediction[0][0] * const.ScreenSize()
                outputy = prediction[0][1] * const.ScreenSize()
            #return actions.FunctionCall(function_id, args)

                return actions.FUNCTIONS.Attack_screen("now", (outputx,outputy))
        else: 
            marine = self.get_units_by_type(obs, units.Terran.Marine)
            if len(marine) > 0: 
                return actions.FUNCTIONS.select_point('select_all_type', (marine[0].x,
                                                                    marine[0].y))


class RandomAgent(base_agent.BaseAgent):
  """A random agent for starcraft."""
  inputActionPairs = np.empty([1])
  prevUnits = 0

  def hasGameReset(self,obs,prevUnits):
      numUnits = len(obs.observation.feature_units)
      if (numUnits > prevUnits):
          self.prevUnits = numUnits
          print("GameReset!")
          return True
      else: 
          self.prevUnits = numUnits
          return False

  def getSimplifiedInput(self, obs):
    #31x31 Simplified
    input = obs.observation.feature_minimap[5]
    stencil = obs.observation.feature_minimap[3]
    newInput = numpy.zeros((const.InputSize(),const.InputSize()),int)
    counterx = 0
    countery = 0
    for numy, y in enumerate(stencil):
        for numx, x in enumerate(y): 
            if (x == 1):
                newInput[countery][counterx] = input[numy][numx]
                counterx+=1
            if (counterx == const.InputSize()):
                countery+=1
                counterx=0
    return newInput
    

  def step(self, obs):
    super(RandomAgent, self).step(obs)

    if (self.hasGameReset(obs,self.prevUnits)):
        self.steps = 0 
        self.reward = 0
    #input = self.getSimplifiedInput(obs)
    #for x in input:
    #        output = ""
    #        for i in x:
    #            output+=str(i)
    #            output+=""
    #        print(output)
    #print("\n")
    
    #for i in range(0,15):
    #    for arg in self.action_spec.functions[i].args:
    #        print(arg.sizes)
    #        print('\n')



    #number between 0-14 inclusive 

    output1 = random.uniform(0,1)*10
    #output2 = 0.7
    #output3 = 0.4
    #oa = [1,2,3]
    #newouput1 = numpy.round(output1*(len(obs.observation.available_actions)),0)
    #n = 0
    #newoutput2 = [[oa[n+size] for size in arg.sizes] 
    #        for arg in self.action_spec.functions[newouput1].args]
    #print(newoutput2)
    #for size in arg.sizes:
    #    for arg in self.action_spec.functions[function_id].args:
    #        newoutput2.append(size,arg)

    #print(newoutput2)
    #while True:
    function_id = 0
    #attack-screen
    if (output1 < 5):
        if (12 in obs.observation.available_actions):
            function_id = 12
            print("Attack")
    ##select rect
    #elif (output1 < 4):
    #    if (3 in obs.observation.available_actions):
    #        function_id = 3
    #        print("Select rect")
    #select army
    #elif (output1 < 6):
    #    if (7 in obs.observation.available_actions):
    #        function_id = 7
    #        print("All units")
    #Move-screen
    elif (output1 < 10):
        if (331 in obs.observation.available_actions):
            function_id = 331
            print("Select point")
    ##stop
    #elif (output1 < 10):
    #    if (274 in obs.observation.available_actions):
    #        function_id = 274
    #        print("Stop")

    #numpy.random.choice(obs.observation.available_actions)
    #for x in obs.observation.available_actions:
    #    print(x)
    #for x in self.action_spec.functions:
    #    print(x)
    args = [[numpy.random.randint(0, size) for size in arg.sizes] 
            for arg in self.action_spec.functions[function_id].args]

        #if ((str)(np.array(args).shape) == "(2,)"):
        #    break
    #numpy.array(args)
    if (function_id == 5):
        args[1] = [48]


    #args[0] = [(numpy.random.randint(0, 3))]
    #args[1][0] = numpy.random.randint(0, 84)
    #args[1][1] = numpy.random.randint(0, 84)
    #args[2][0] = 0
    #args[2][1] = 0
    print("Function ID: {} Args: {}".format(function_id,args))
    return actions.FunctionCall(function_id, args)
