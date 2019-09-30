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
from pysc2.lib.actions import FUNCTIONS
from Translator import Translator
from TranslateOutputs import translate
from Constants import const
from DeepNetwork import get_closest_ability
import numpy as np
import keras as ks
import math
from absl import app

class Overmind(base_agent.BaseAgent):
    loaded = False
    zergSpeed = False
    homeHatch = None
    # #One-time setup
    # def __init__(self):
    #     super(Overmind, self).__init__()


    def get_units_by_type(self, obs, unit_type):
        if (unit_type == units.Zerg.Hatchery):
            return [unit for unit in obs.observation.raw_units
                    if unit.unit_type == units.Zerg.Hatchery or
                    unit.unit_type == units.Zerg.Hive or
                    unit.unit_type == units.Zerg.Lair]
        else:
            return [unit for unit in obs.observation.raw_units
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

    def get_buildings(self, obs, building, in_progress=True):
        if in_progress:
            if (building == units.Zerg.Hatchery):
                hatcheries = [unit for unit in obs.observation.raw_units
                              if unit.unit_type == units.Zerg.Lair]
                hatcheries += [unit for unit in obs.observation.raw_units
                               if unit.unit_type == units.Zerg.Hive]
                hatcheries += [unit for unit in obs.observation.raw_units
                               if unit.unit_type == units.Zerg.Hatchery]
                return hatcheries
            return [unit for unit in obs.observation.raw_units
                    if unit.unit_type == building]
        else:
            if (building == units.Zerg.Hatchery):
                hatcheries = [unit for unit in obs.observation.raw_units
                              if unit.unit_type == units.Zerg.Lair and unit.build_progress == 100]
                hatcheries += [unit for unit in obs.observation.raw_units
                              if unit.unit_type == units.Zerg.Hive and unit.build_progress == 100]
                hatcheries += [unit for unit in obs.observation.raw_units
                              if unit.unit_type == units.Zerg.Hatchery and unit.build_progress == 100]
                return hatcheries
            return [unit for unit in obs.observation.raw_units
                              if unit.unit_type == building and unit.build_progress == 100]

    def build_unit(self, obs, unit_type):
        # Queen
        if (unit_type == units.Zerg.Queen):
            if (self.unit_type_is_selected(obs, units.Zerg.Hatchery) or
                    self.unit_type_is_selected(obs, units.Zerg.Hive) or
                    self.unit_type_is_selected(obs, units.Zerg.Lair)):
                if (self.can_do(obs, FUNCTIONS.Train_Queen_quick.id)):
                    return FUNCTIONS.Train_Queen_quick('now')
            else:
                bases = self.get_buildings(obs, units.Zerg.Hatchery, False)
                return FUNCTIONS.select_point('select_all_type', (bases[-1].x*2,
                                                                          bases[-1].y*2))
        if (not self.unit_type_is_selected(obs, units.Zerg.Larva)):
            larva = self.get_units_by_type(obs, units.Zerg.Larva)
            if len(larva) > 0:
                x = larva[-1].x
                y = larva[-1].y
                return FUNCTIONS.select_point('select_all_type', (larva[-1].x*2,
                                                                      larva[-1].y*2))
        # Drone
        if (unit_type == units.Zerg.Drone):
            if (self.can_do(obs, FUNCTIONS.Train_Drone_quick.id)):
                return FUNCTIONS.Train_Drone_quick('now')
        # Overlord
        if (unit_type == units.Zerg.Overlord):
            if (self.can_do(obs, FUNCTIONS.Train_Overlord_quick.id)):
                return FUNCTIONS.Train_Overlord_quick('now')
        # Zergling
        if (unit_type == units.Zerg.Zergling):
            if (self.can_do(obs, FUNCTIONS.Train_Zergling_quick.id)):
                return FUNCTIONS.Train_Zergling_quick('now')
        # Roach
        if (unit_type == units.Zerg.Roach):
            if (self.can_do(obs, FUNCTIONS.Train_Roach_quick.id)):
                return FUNCTIONS.Train_Roach_quick('now')
        return FUNCTIONS.no_op()

    def build_building(self, obs, building_type):
        if (not self.unit_type_is_selected(obs, units.Zerg.Drone)):
            drones = self.get_units_by_type(obs, units.Zerg.Drone)
            return FUNCTIONS.select_point('select_all_type', (drones[-1].x*2,
                                                              drones[-1].y*2))
        hatcheries = self.get_units_by_type(obs, units.Zerg.Hatchery)
        if (building_type == units.Zerg.Hatchery):
            if (self.can_do(obs, FUNCTIONS.Build_Hatchery_screen.id)):
                if hatcheries[0].y*2 < 200:
                    if len(hatcheries) == 1:
                        return FUNCTIONS.Build_Hatchery_screen("now", [58, 138])
                    else:
                        return FUNCTIONS.Build_Hatchery_screen("now", [122, 78])
                else:
                    if len(hatcheries) == 1:
                        return FUNCTIONS.Build_Hatchery_screen("now", [292, 244])
                    else:
                        return FUNCTIONS.Build_Hatchery_screen("now", [228, 304])

        if (building_type == units.Zerg.Extractor):
            if (self.can_do(obs, FUNCTIONS.Build_Extractor_screen.id)):
                if hatcheries[0].y < 100:
                    xy = 0
                else:
                    xy = 350
                closestExtractor = self.get_closest_unit_to_pos(obs, xy, xy, units.Neutral.VespeneGeyser)
                return FUNCTIONS.Build_Extractor_screen("now", [closestExtractor.x*2, closestExtractor.y*2])

        if (building_type == units.Zerg.SpawningPool):
            if (self.can_do(obs, FUNCTIONS.Build_SpawningPool_screen.id)):
                if hatcheries[0].y*2 > 200:
                    offset = -10
                else:
                    offset = 10
                return FUNCTIONS.Build_SpawningPool_screen("now", [self.homeHatch.x*2, (self.homeHatch.y*2) + offset])

        return FUNCTIONS.no_op()

    def upgrade(self, obs, upgrade):
        if not self.unit_type_is_selected(units.Zerg.SpawningPool):
            spawningPool = self.get_buildings(obs, units.Zerg.SpawningPool)
            return FUNCTIONS.select_point('select_all_type', (spawningPool[-1].x*2,
                                                              spawningPool[-1].y*2))
        else:
            return FUNCTIONS.Research_ZerglingMetabolicBoost_quick("now")
        return FUNCTIONS.no_op()

    def redestribute_workers(self, obs):
        hatcheries = self.get_buildings(obs, units.Zerg.Hatchery)
        extractors = self.get_buildings(obs, units.Zerg.Extractor)
        if not self.unit_type_is_selected(obs, units.Zerg.Drone) and obs.observation.player.idle_worker_count > 0:
            for worker in self.get_units_by_type(obs, units.Zerg.Drone):
                if worker["order_length"] == 0:
                    return FUNCTIONS.select_rect([0], (worker.x*2-2, worker.y*2-2), (worker.x*2+2, worker.y*2+2))

        for hatch in hatcheries:
            # If excess harvesters
            if hatch.assigned_harvesters - hatch.ideal_harvesters > 0:
                # if we haven't selected a drone or just told a drone to move -> select drone
                if not self.unit_type_is_selected(obs, units.Zerg.Drone) or 451 in obs.observation.last_actions:
                    drone = self.get_closest_unit_to_pos(obs, hatch.x, hatch.y, units.Zerg.Drone)
                    return FUNCTIONS.select_point([0], (drone.x*2, drone.y*2))
                else:
                    for extractor in extractors:
                        if extractor.assigned_harvesters - extractor.ideal_harvesters < 0:
                            return FUNCTIONS.Smart_screen([0], (extractor.x * 2, extractor.y * 2))
            # Not excess harvesters -> move drone to this place
            else:
                if self.unit_type_is_selected(obs, units.Zerg.Drone) and 451 not in obs.observation.last_actions:
                    mineral = self.get_closest_unit_to_pos(obs, hatch.x, hatch.y, units.Neutral.MineralField)
                    return FUNCTIONS.Smart_screen([0], (mineral.x*2, mineral.y*2))
        return self.build_unit(obs, units.Zerg.Drone)

    def get_closest_unit_to_pos(self, obs, x, y, type):
        closest_unit = None
        closestDist = 10000
        for unit in obs.observation.raw_units:
            if unit.unit_type == type:
                dist = math.sqrt((unit.x - x) ** 2 + (unit.y - y) ** 2)
                if dist < closestDist:
                    closest_unit = unit
                    closestDist = dist
        return closest_unit

    def macro(self, obs):
        if obs.observation.player.idle_worker_count > 0:
            return self.redestribute_workers(obs)
        if self.unit_type_is_selected(obs, units.Zerg.Overlord):
            if 451 not in obs.observation.last_actions:
                hatcheries = self.get_buildings(obs, units.Zerg.Hatchery)
                if hatcheries[0].y < 100:
                    return FUNCTIONS.Smart_screen("now", (0, 0))
                else:
                    return FUNCTIONS.Smart_screen("now", (350, 350))

        dronenum = obs.observation.player['food_workers']
        if (obs.observation.player['food_cap'] - obs.observation.player['food_used'] < 2):
            return self.build_unit(obs, units.Zerg.Overlord)
        # These buildings are essential and cannot be skipped
        if dronenum > 17:
            if len(self.get_buildings(obs, units.Zerg.Hatchery)) == 1:
                return self.build_building(obs, units.Zerg.Hatchery)
            if len(self.get_buildings(obs, units.Zerg.Extractor)) == 0:
                return self.build_building(obs, units.Zerg.Extractor)
            if len(self.get_buildings(obs, units.Zerg.SpawningPool)) == 0:
                return self.build_building(obs, units.Zerg.SpawningPool)
        if dronenum == 19:
            return self.redestribute_workers(obs)
            print("Drones in gas")
        if dronenum == 20 and len(obs.observation.build_queue) < 2 and len(self.get_units_by_type(obs, units.Zerg.Queen)) > 0:
            return self.build_unit(obs, units.Zerg.Queen)
        if dronenum == 27:
            # if not self.zergSpeed:
            #     return self.upgrade(1)
            if len(self.get_units_by_type(obs, units.Zerg.Zergling)) < 100:
                return self.build_unit(obs, units.Zerg.Zergling)
        if dronenum == 28 and len(self.get_buildings(obs, units.Zerg.Hatchery)) == 2:
            return self.build_building(obs, units.Zerg.Hatchery)
        if dronenum == 29:
            return self.redestribute_workers(obs)
        if dronenum == 40:
            if len(self.get_buildings(obs, units.Zerg.Extractor)) < 4:
                return self.build_building(obs, units.Zerg.Extractor)
            if len(self.get_buildings(obs, units.Zerg.SporeCrawler) < 3):
                return self.build_building(obs, units.Zerg.SporeCrawler)
            if len(self.get_buildings(obs, units.Zerg.RoachWarren) < 1):
                return self.build_building(obs, units.Zerg.RoachWarren)
            if len(self.get_buildings(obs, units.Zerg.EvolutionChamber) < 2):
                return self.build_building(obs, units.Zerg.EvolutionChamber)
        if dronenum == 41:
            return self.redestribute_workers(obs)
        if dronenum > 60:
            #ALL UPGRADES
            return self.build_unit(obs, units.Zerg.Roach)

        return self.build_unit(obs, units.Zerg.Drone)


    #Each step
    def step(self, obs):
        super(Overmind, self).step(obs)

        if (not self.loaded):
            self.loaded = True
            self.homeHatch = self.get_buildings(obs, units.Zerg.Hatchery)[0]
            self.model = ks.models.load_model("C:\\Models\\Conv2D-LSTM")
            return FUNCTIONS.move_camera([const.MiniMapSize().x / 2, const.MiniMapSize().y / 2])

        # If nothing to macro
        # if obs.observation.player['food_army'] < 30:
        #     function = self.macro(obs)
        #     print(function)
        #     return function

        T = Translator()

        tFeatureLayers = T.translate_feature_layers(T.crop_feature_layers(obs.observation.feature_screen[0],
                                                                       obs.observation.feature_screen,
                                                                       const.ScreenSize().x, const.ScreenSize().y))

        featureLayers = (np.moveaxis((np.array(tFeatureLayers)), 0, 2)).reshape(-1, const.ScreenSize().x, const.ScreenSize().y, 12)
        prediction = self.model.predict(featureLayers)[0]
        action = translate(obs, prediction)
        print(action)
        if (action[0] in obs.observation.available_actions):
            print(action[0])
            return actions.FunctionCall(action[0], action[1:])
        else:
            print("No-op, Switching to Macro:")
            function = self.macro(obs)
            print(function)
            return function