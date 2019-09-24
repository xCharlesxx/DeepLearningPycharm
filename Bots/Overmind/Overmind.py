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

class Overmind(base_agent.BaseAgent):
    loaded = False
    buildings = []
    #One-time setup
    def __init__(self):

        super(Overmind, self).__init__()
        self.model = ks.models.load_model("C:\\PycharmProjects\\models\\Conv2D-Full")

    def get_units_by_type(self, obs, unit_type):
        if (unit_type == units.Zerg.Hatchery):
            return [unit for unit in obs.observation.feature_units
                    if unit.unit_type == units.Zerg.Hatchery or
                    unit.unit_type == units.Zerg.Hive or
                    unit.unit_type == units.Zerg.Lair]
        else:
            return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.multi_select) > 0):
            if (obs.observation.multi_select[0].unit_type == unit_type):
                return True
        elif (len(obs.observation.single_select) > 0):
            if (obs.observation.single_select[0].unit_type == unit_type):
                return True

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def get_buildings(self, obs, building):
        if (building == units.Zerg.Hatchery):
            hatcheries = units.Zerg.Lair in self.buildings
            hatcheries += units.Zerg.Hive in self.buildings
            hatcheries += units.Zerg.Hatchery in self.buildings
            return hatcheries
        return building in self.buildings

    def build_unit(self, obs, unit_type):
        # Queen
        if (unit_type == units.Zerg.Queen):
            if (self.unit_type_is_selected(obs, units.Zerg.Hatchery) or
                    self.unit_type_is_selected(obs, units.Zerg.Hive) or
                    self.unit_type_is_selected(obs, units.Zerg.Lair)):
                if (self.can_do(obs, actions.FUNCTIONS.Train_Queen_quick.id)):
                    return actions.FUNCTIONS.Train_Queen_quick('now')
            else:
                bases = self.get_buildings(obs, units.Zerg.Hatchery)
                return actions.FUNCTIONS.select_point('select_all_type', (bases[-1].x,
                                                                          bases[-1].y))
        if (not self.unit_type_is_selected(obs, units.Zerg.Larva)):
            if (self.can_do(obs, actions.FUNCTIONS.select_larva.id)):
                return actions.FUNCTIONS.select_larva()
        # Drone
        if (unit_type == units.Zerg.Drone):
            if (self.can_do(obs, actions.FUNCTIONS.Train_Drone_quick.id)):
                return actions.FUNCTIONS.Train_Drone_quick('now')
        # Overlord
        if (unit_type == units.Zerg.Overlord):
            if (self.can_do(obs, actions.FUNCTIONS.Train_Overlord_quick.id)):
                return actions.FUNCTIONS.Train_Overlord_quick('now')
        # Zergling
        if (unit_type == units.Zerg.Zergling):
            if (self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id)):
                return actions.FUNCTIONS.Train_Zergling_quick('now')
        # Roach
        if (unit_type == units.Zerg.Roach):
            if (self.can_do(obs, actions.FUNCTIONS.Train_Roach_quick.id)):
                return actions.FUNCTIONS.Train_Roach_quick('now')

    #Each step
    def step(self, obs):
        super(Overmind, self).step(obs)

        if (not self.loaded):
            self.loaded = True
            self.model = ks.models.load_model("C:\\PycharmProjects\\models\\Conv2D-Full")
            self.buildings.append(self.get_units_by_type(obs, units.Zerg.Hatchery))
            return actions.FUNCTIONS.move_camera([const.MiniMapSize().x / 2, const.MiniMapSize().y / 2])

        return self.build_unit(obs, units.Zerg.Drone)

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