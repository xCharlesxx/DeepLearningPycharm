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
from TranslateOutputs import translate
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
     self.model = ks.models.load_model("G:\\Models\\Conv2D-80k")

    #Each step
    def step(self, obs):
        super(Overmindx00, self).step(obs)

        if (not self.loaded):
            self.loaded = True
            self.model = ks.models.load_model("G:\\Models\\Conv2D-80k")
            return actions.FUNCTIONS.move_camera([const.MiniMapSize().x / 2, const.MiniMapSize().y / 2])

        T = Translator()

        tFeatureLayers = T.translate_feature_layers(T.crop_feature_layers(obs.observation.feature_screen[0],
                                                                       obs.observation.feature_screen,
                                                                       const.ScreenSize().x, const.ScreenSize().y))

        featureLayers = (np.moveaxis((np.array(tFeatureLayers)), 0, 2)).reshape(-1, const.ScreenSize().x, const.ScreenSize().y, 12)
        prediction = self.model.predict(featureLayers)[0]
        action = int(translate(obs, prediction))
        print(action)
        if (action[0] in obs.observation.available_actions):
            print(action[0])
            return actions.FunctionCall(action[0], action[1:])
        else:
            print("No-op")
            return actions.FUNCTIONS.no_op()