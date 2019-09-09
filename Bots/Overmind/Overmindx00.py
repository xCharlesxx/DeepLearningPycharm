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
from Translator import Translator
from Constants import const
from DeepNetwork import get_closest_ability
import numpy as np
import keras as ks
from absl import app

class Overmindx00(base_agent.BaseAgent):
    loaded = False
    #One-time setup
    def __init__(self):
     super(Overmindx00, self).__init__()
     self.model = ks.models.load_model("C:\\Users\\LeoCharlie\\PycharmProjects\\DeepLearning\\Models\\Conv2D-Attempt1")
    def select_point(self, type, x, y):
        act = [2]
        act.append([int(type)])
        act.append([x * const.ScreenSize().x, y * const.ScreenSize().x])
        return act
    def select_rect(self, x1, y1, x2, y2):
        act = [3]
        act.append([int(0)])
        act.append([x1 * const.ScreenSize().x, y1 * const.ScreenSize().x])
        act.append([x2 * const.ScreenSize().x, y2 * const.ScreenSize().x])
        return act
    def smart_screen(self, x, y):
        act = [451]
        act.append([int(0)])
        act.append([x * const.ScreenSize().x, y * const.ScreenSize().x])
        return act
    def attack_point(self, x, y):
        act = [12]
        act.append([int(0)])
        act.append([x * const.ScreenSize().x, y * const.ScreenSize().x])
        return act
    def hold_pos(self):
        act = [453]
        act.append([int(0)])
        return act
    def select_army(self):
        act = [7]
        act.append([int(0)])
        return act
    def use_ability(self, x, y, type, available):
        act = [get_closest_ability(type, available)]
        act.append([int(0)])
        act.append([x * const.ScreenSize().x, y * const.ScreenSize().x])
        return act
    #Each step
    def step(self, obs):
        super(Overmindx00, self).step(obs)

        if (not self.loaded):
            self.loaded = True
            self.model = ks.models.load_model("C:\\Users\\LeoCharlie\\PycharmProjects\\DeepLearning\\Models\\Conv2D-80k")
            return actions.FUNCTIONS.move_camera([const.MiniMapSize().x / 2, const.MiniMapSize().y / 2])

        T = Translator()

        tFeatureLayers = T.translate_feature_layers(T.crop_feature_layers(obs.observation.feature_screen[0],
                                                                       obs.observation.feature_screen,
                                                                       const.ScreenSize().x, const.ScreenSize().y))

        featureLayers = (np.moveaxis((np.array(tFeatureLayers)), 0, 2)).reshape(-1, const.ScreenSize().x, const.ScreenSize().y, 12)
        prediction = self.model.predict(featureLayers)[0]
        choice = prediction[:7]
        m = max(choice)
        choice = [i for i, j in enumerate(choice) if j == m]
        choices = {
         0: self.select_point(prediction[7], prediction[8], prediction[9]),
         1: self.select_rect(prediction[8], prediction[9], prediction[10], prediction[11]),
         2: self.smart_screen(prediction[8], prediction[9]),
         3: self.attack_point(prediction[8], prediction[9]),
         4: self.hold_pos,
         5: self.select_army,
         6: self.use_ability(prediction[8], prediction[9], prediction[12], obs.observation.available_actions)
        }
        action = choices.get(choice[0])
        #action = int(action)
        print(action)
        if (action[0] in obs.observation.available_actions):
            print(action[0])
            #args = action[1:]
            # args[1] = [round(x) for x in args[1]]
            # args[1] = [int(x) for x in args[1]]
            #args[0][0] = int(args[0][0])
            #print(args)
            return actions.FunctionCall(action[0], action[1:])
        else:
            print("No-op")
            return actions.FUNCTIONS.no_op()