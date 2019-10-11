from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import pysc2
import numpy
import Bots

from Bots.MoveToBeacon import MoveToBeacon, GenerateMoveToBeaconTestData
from Bots.Overmind.Overmind import Overmind
from DeepNetwork import build_knet, build_transformer, build_LSTM, train_LSTM
from Bots.DefeatEnemies import RandomAgent, DefeatEnemies
from pysc2Replay.ObserverAgent import NothingAgent
from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units
from Constants import const
from keras.backend.tensorflow_backend import set_session
import gc
from absl import app

def main(unused_argv):
    #build_knet()
    #build_transformer()
    #build_LSTM()
    # Dynamically grow the memory used on GPU
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # set_session(sess)
    # for i in range(0, 100):
    train_LSTM()

    #transform_replay
    #Agent
    agent = Overmind()
    try: 
        while True:
            with sc2_env.SC2Env(False,
                map_name = 'KingsCove',
                players= [
                        sc2_env.Agent(sc2_env.Race.zerg),
                        sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)
                         ],
                agent_interface_format=features.AgentInterfaceFormat(
                    #What resolution the player sees the world at 
                    feature_dimensions=features.Dimensions(screen=const.ScreenSize(), minimap=const.MiniMapSize()),
                    #More indepth unit information
                    use_raw_units=True,
                    use_camera_position=True,
                    #Increase camera size to encompass whole map
                    camera_width_world_units=round(const.WorldSize().x)),
                #Steps default is 8 per frame (168APM) (16 = 1 second)
                step_mul=16, # 175,
                #Max steps per game (0 is infinite)
                game_steps_per_episode=0,
                #visualize pysc2 input layers 
                visualize=False,
                #Play-back-time
                realtime=False,
                #Fog of War
                #disable_fog=False,
           ) as env:
                run_loop.run_loop([agent], env)
    except KeyboardInterrupt:
        pass



if __name__ == "__main__":
      app.run(main)