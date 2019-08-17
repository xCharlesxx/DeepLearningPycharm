from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import pysc2
import numpy

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units


from absl import app

class Overmindx00(base_agent.BaseAgent):
    #One-time setup
    def __init__(self):
     super(Overmindx00, self).__init__()
     

     #Before each game 
     def reset(self):
         super(Overmindx00, self).reset()
         
     #Each step
     def step(self, obs):
         super(Overmindx00, self).step(obs) 

         #Read state from obs and ALWAYS act
         return actions.FUNCTIONS.no_op()