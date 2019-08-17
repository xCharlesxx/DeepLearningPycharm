#!/usr/bin/env python

import pickle
import numpy as np
import math

from pysc2.lib import actions as sc_action
from pysc2.lib import static_data
from pysc2.agents import base_agent
from Constants import const
from Translator import Translator

class ObserverAgent(base_agent.BaseAgent):
    action_dict = {}
    cam_pos_offset = [0, 0]
    def __init__(self):
        self.states = []
        self.count = 0;
        self.action_dict = {
                #select point
                2: self.select_point,
                #select rect 
                3: self.double_select_point,
                #smart minimap 
                452: self.single_select_point,
                #attack minimap 
                13: self.single_select_point,
                17: self.single_select_point,
                #smart screen 
                451: self.single_select_point,
                #attack screen 
                12: self.single_select_point,
                14: self.single_select_point,
                16: self.single_select_point,
                #Inject larve
                204: self.single_select_point,
                #Burrow down
                103: self.single_q,
                #Burrow up
                117: self.single_q,
                #Blinding cloud 
                179: self.single_select_point,
                #Caustic spray 
                184: self.single_select_point,
                #Contaminate 
                188: self.single_select_point,
                #Corrosive bile 
                189: self.single_select_point,
                #Explode 
                191: self.single_q,
                #fungal growth 
                194: self.single_select_point,
                #infested terrans 
                203: self.single_select_point,
                #transfuse 
                242: self.single_select_point,
                #neural parasite 
                212: self.single_select_point,
                #parasidic bomb 
                215: self.single_select_point,
                #spawn changeling
                228: self.single_q,
                #spawn locusts
                229: self.single_select_point,
                #viper consume 
                243: self.single_select_point,
                #hold position quick 
                274: self.single_q,
                #stop quick
                453: self.single_q,
                #select army 
                7: self.single_q
                    }

    def calculate_offset(self, height_map): 
        for numy, y in enumerate(height_map):
            for numx, x in enumerate(y): 
                if (x != 0):
                    self.cy_cx = [numy, numx]
                    return

    def stencil(self, _stencil, _raw_list, _new_width, _new_height):
        input = _raw_list
        stencil = _stencil
        newInput = np.zeros((_new_height,_new_width),int)
        counterx = 0
        countery = 0
        for numy, y in enumerate(stencil):
            for numx, x in enumerate(y): 
                if (x != 0):
                    newInput[countery][counterx] = input[numy][numx]
                    counterx+=1
                if (counterx == _new_width):
                    countery+=1
                    counterx=0
        return newInput

    def select_point(self, args):
        return list(([int(args[0][0])], [(args[1][0]/2) + self.cam_pos_offset[0], (args[1][1]/2) + self.cam_pos_offset[1]]))

    def single_select_point(self, args):
        return list(([[0], (args[1][0]/2) + self.cam_pos_offset[0], (args[1][1]/2) + self.cam_pos_offset[1]]))

    def double_select_point(self, args):
        ##TODO if any of the values are the same
        if (args[1] == args[2]):
            return "Unknown"
        list = [0]
        list.append([(args[1][0]/2) + self.cam_pos_offset[0], (args[1][1]/2) + self.cam_pos_offset[1]])
        list.append([(args[2][0]/2) + self.cam_pos_offset[0], (args[2][1]/2) + self.cam_pos_offset[1]])
        return list
    def single_q(self, args):
        return [0]
    def default(self, args):
        return "Unknown"

    def extract_args(self, id, args):
        if not args: 
            return []

        #Burrow down for various units translate to quick burrow up
        if (id > 103 and id < 117):
            id = '103'

        #Same for burrow up
        if (id > 117 and id < 140):
            id = '117'

        if (id in [452, 13, 17]):
            return [0, [args[1][0] / 2, args[1][1] / 2]]

        func = self.action_dict.get(id, self.default)
        output = func(args)
        if output is "Unknown":
            return output
        output.insert(0, id)
        return output


    def step(self, time_step, info, act):
        self.cam_pos_offset = [time_step.observation.camera_position[0] - (const.WorldSize('x')/2), time_step.observation.camera_position[1] - (const.WorldSize('y')/2)]
        #print(time_step.observation.camera_position)
        #print(self.cam_pos_offset)
        #print(time_step.observation.camera_position - self.cam_pos_offset)
        print (act)
        state = {"action": [self.extract_args(int(act.function), act.arguments)]}
        if ("Unknown" in state["action"][0]):
            return 0
        print(state["action"])
        height_map = time_step.observation.feature_screen[0]
        #Remove all Zero lines 
        height_map = height_map[~np.all(height_map == 0, axis=1)]
        #Remove all Zeros in remaining lines
        height_map = [x[x != 0] for x in height_map]

        height = 0
        width = 0
        for x in height_map:
            output = ""
            width = 0
            for i in x:
                output+=str(i)
                output+=""
                width += 1
            #print(output)
            height += 1
        #print("\n")
        print("W: {} H: {}".format(width, height))
        # print("width: ")
        # print(width)
        # print("height: ")
        # print(height)
        #print("\n")
        #return
        # T = Translator()
        #
        # tFeatureLayers = T.translate_feature_layers(T.crop_feature_layers(time_step.observation.feature_screen[0],
        #                                                                  time_step.observation.feature_screen,
        #                                                                 width, height))
        #
        # state["feature_layers"] = tFeatureLayers

        # for y in tFeatureLayers:
        #    for x in y:
        #        output = ""
        #        for i in x:
        #            output+=str(i)
               #print(output)
           #print("\n")
        

    #[0 "height_map", 
    # 1 "visibility_map", 
    # 2 "creep", 
    # 3 "power", 
    # 4 "player_id",
    # 5 "player_relative", 
    # 6 "unit_type", 
    # 7 "selected", 
    # 8 "unit_hit_points",
    # 9 "unit_hit_points_ratio", 
    # 10"unit_energy", 
    # 11"unit_energy_ratio", 
    # 12"unit_shields",
    # 13"unit_shields_ratio", 
    # 14"unit_density", 
    # 15"unit_density_aa", 
    # 16"effects"]

        return state


class NothingAgent(base_agent.BaseAgent):
    def step(self, obs):
        #print(obs.observation.available_actions)
        height = 0
        for x in obs.observation.feature_screen[0]:
            output = ""
            width = 0
            for i in x:
                output+=str(i)
                output+=""
                width += 1
            print(output)
            height += 1
        print("\n")
        if (1 in obs.observation.available_actions):
            return sc_action.FUNCTIONS.move_camera([const.MiniMapSize()/2,const.MiniMapSize()/2])
        return sc_action.FUNCTIONS.no_op()