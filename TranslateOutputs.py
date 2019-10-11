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


def select_point(type, x, y):
    act = [2]
    act.append([int(type)])
    act.append([x * const.ScreenSize().x, y * const.ScreenSize().x])
    return act
def select_rect(x1, y1, x2, y2):
    act = [3]
    act.append([int(0)])
    act.append([x1 * const.ScreenSize().x, y1 * const.ScreenSize().x])
    act.append([x2 * const.ScreenSize().x, y2 * const.ScreenSize().x])
    return act
def smart_screen(x, y):
    act = [451]
    act.append([int(0)])
    act.append([x * const.ScreenSize().x, y * const.ScreenSize().x])
    return act
def attack_point(x, y):
    act = [12]
    act.append([int(0)])
    act.append([x * const.ScreenSize().x, y * const.ScreenSize().x])
    return act
def hold_pos():
    act = [453]
    act.append([int(0)])
    return act
def select_army():
    act = [7]
    act.append([int(0)])
    return act
def use_ability(x, y, type, available):
    act = [get_closest_ability(type, available)]
    if act == 0:
        return [0]
    act.append([int(0)])
    act.append([x * const.ScreenSize().x, y * const.ScreenSize().x])
    return act
def translate(obs, prediction):
    choice = prediction[0]
    m = max(choice)
    choice = [i for i, j in enumerate(choice) if j == m]
    choices = {
     0: select_point(prediction[1][0], prediction[1][1], prediction[1][2]),
     #1: select_rect(prediction[1][1], prediction[1][2], prediction[1][3], prediction[1][4]),
     1: smart_screen(prediction[1][1], prediction[1][2]),
     2: attack_point(prediction[1][1], prediction[1][2]),
     #4: hold_pos,
     3: select_army,
     4: use_ability(prediction[1][1], prediction[1][2], prediction[1][3], obs.observation.available_actions)
    }
    return choices.get(choice[0])