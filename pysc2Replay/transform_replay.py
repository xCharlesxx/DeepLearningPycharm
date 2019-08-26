from pysc2.lib import features, point, remote_controller, actions, units
from absl import app, flags
from pysc2.env.environment import TimeStep, StepType
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb
import importlib
import glob
from random import randint
import pickle
from multiprocessing import Process
from tqdm import tqdm
import math
import random
import numpy as np
import multiprocessing
import sys, os, csv
from Constants import const

cpus = multiprocessing.cpu_count()

FLAGS = flags.FLAGS
FLAGS(sys.argv)
flags.DEFINE_string("replays", "C:\Program Files (x86)\StarCraft II\Replays\Test", "Path to the replay files.")
flags.DEFINE_string("agent", "ObserverAgent.ObserverAgent", "Path to an agent.")
flags.DEFINE_integer("procs", cpus, "Number of processes.", lower_bound=1)
flags.DEFINE_integer("start", 0, "Start at replay no.", lower_bound=0)
flags.DEFINE_integer("batch", 1, "Size of replay batch for each process", lower_bound=1, upper_bound=512)

class Parser: #612
    screen_size_px=(const.WorldSize().x*4, const.WorldSize().x*4)
    minimap_size_px=(const.WorldSize().x*4, const.WorldSize().x*4)
    camera_width = const.WorldSize().x * 2

    def __init__(self,
                 replay_file_path,
                 agent,
                 player_id=1,
                 discount=1.):
                 #frames_per_game=1):

        print("Parsing " + replay_file_path)

        self.replay_file_name = replay_file_path.split("\\")[-1].split(".")[0]
        self.agent = agent
        self.discount = discount
        #self.frames_per_game = frames_per_game

        self.run_config = run_configs.get()
        versions = self.run_config.get_versions()

        self.sc2_proc = self.run_config.start()
        self.controller = self.sc2_proc.controller

        replay_data = self.run_config.replay_data(replay_file_path)
        ping = self.controller.ping()
        self.info = self.controller.replay_info(replay_data)
        if not self._valid_replay(self.info, ping):
            raise Exception("{} is not a valid replay file!".format(replay_file_path))

        _screen_size_px = point.Point(*self.screen_size_px)
        _minimap_size_px = point.Point(*self.minimap_size_px)
        interface = sc_pb.InterfaceOptions(
            feature_layer=sc_pb.SpatialCameraSetup(width=self.camera_width),# crop_to_playable_area=True),
                                                    show_cloaked=True, raw=True)#, raw_affects_selection=True,raw_crop_to_playable_area=True)
        _screen_size_px.assign_to(interface.feature_layer.resolution)
        _minimap_size_px.assign_to(interface.feature_layer.minimap_resolution)

        map_data = None
        if self.info.local_map_path:
            map_data = self.run_config.map_data(self.info.local_map_path)

        self._episode_length = self.info.game_duration_loops
        self._episode_steps = 0

        self.controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id))

        self._state = StepType.FIRST

    @staticmethod
    def _valid_replay(info, ping):
        """Make sure the replay isn't corrupt, and is worth looking at."""
        if (info.HasField("error")):
            print("Replay has field: Error")
            return False
        if (info.base_build != ping.base_build): # different game version
           print("Build Mismatch:\nBase: {}\nReplay: {}".format(ping.base_build, info.base_build))
           return False 
        if (info.game_duration_loops < 10):
            print("Replay not long enough, loops: {}".format(info.game_duration_loops))
            return False
        if (len(info.player_info) != 2):
            print("Replay: Possible corruption")
            # Probably corrupt, or just not interesting.
            return False
#   for p in info.player_info:
#       if p.player_apm < 10 or p.player_mmr < 1000:
#           # Low APM = player just standing around.
#           # Low MMR = corrupt replay or player who is weak.
#           return False
        return True

    def start(self):
        print("Hello we are in Start")
        step_mul = 1
        _features = features.features_from_game_info(self.controller.game_info(), use_camera_position=True)
        #print("world_tl_to_world_camera_rel: {}\n\nworld_to_feature_screen_px: {}\n\nworld_to_world_tl: {}".format(_features._world_tl_to_world_camera_rel,
        #                                                                              _features._world_to_feature_screen_px,
        #                                                                              _features._world_to_world_tl))
        # _features.init_camera(features.Dimensions(self.screen_size_px, self.minimap_size_px),
        #                       point.Point(*const.WorldSize()),
        #                       self.camera_width)
        packageCounter = 0
        fileName = '../training_data/' + self.replay_file_name + "/" + str(packageCounter) + '.csv'
        npFileName = '../training_data/' + self.replay_file_name + "/" + str(packageCounter) + '.npy'
        npFileNameComp = '../training_data/' + self.replay_file_name + "/" + str(packageCounter)
        dirname = os.path.dirname(fileName)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        while True:
            #Takes one step through the replay
            self.controller.step(step_mul)
            #Converts visual data into abstract data
            obs = self.controller.observe()

            if obs.player_result: # Episide over.
                self._state = StepType.LAST
                print("Episode Over")
                break;
                discount = 0
            else:
                discount = self.discount  
                
            if (len(obs.actions) == 0):
                continue

            agent_obs = _features.transform_obs(obs)
            step = TimeStep(step_type=self._state, reward=0,
                            discount=discount, observation=agent_obs)

            for action in obs.actions:
                for num in self.agent.action_dict.keys():
                    # If action is worth recording
                    if (int(_features.reverse_action(action).function) == num):
                        # Check if the action is on a Micro Unit
                        if (const.IsMicroUnit(agent_obs.single_select) or const.IsMicroUnit(agent_obs.multi_select)):
                            # Record action
                            #print(_features._world_tl_to_world_camera_rel.offset)
                            #self.agent.states.append(self.agent.step(step, self.info, _features.reverse_action(action)))
                            state = self.agent.step(step, self.info, _features.reverse_action(action))
                            if state != 0:
                                npFileNameComp = '../training_data/' + self.replay_file_name + "/" + str(packageCounter)
                                np.savez_compressed(npFileNameComp, action=state["action"],
                                                                    feature_layers=state["feature_layers"])
                                packageCounter += 1
                        break
                        #print("%s: %s" % (len(agent_obs.multi_select), units.Zerg(agent_obs.multi_select[0][0])))
                        #print(action)
                        #print(units.Zerg(agent_obs.single_select[0][0]))

            #self.agent.step(step, self.info, acts)
            #print(_features.reverse_action(obs.actions[0]))
            #print ("+")
            #print(offset)
            #screenpoint = (84, 84)
            #screenpoint = point.Point(*screenpoint)
            
            if obs.player_result:
                break

            self._state = StepType.MID

        # print("Saving data")
        # #print(self.info)
        # #print(self.agent.states)
        # for packageCounter, state in enumerate(self.agent.states):
        #     fileName = '../training_data/' + self.replay_file_name + "/" + str(packageCounter) + '.csv'
        #     npFileName = '../training_data/' + self.replay_file_name + "/" + str(packageCounter) + '.npy'
        #     dirname = os.path.dirname(fileName)
        #     if not os.path.exists(dirname):
        #         os.makedirs(dirname)
        #     outarr = []
        #     outarr.append(state['action'])
        #     for layer in state['feature_layers']:
        #         outarr.append(layer)
        #     np.save(npFileName, outarr)
        #     # with open(fileName, mode='w', newline='') as file:
        #     #     writer = csv.writer(file)
        #     #     # writer.writerow(state['action'])
        #     #     # writer.writerows(state['feature_layers'])
        #     #     for action in state['action']:
        #     #         writer.writerow(action)
        #     #     for layer in state['feature_layers']:
        #     #         writer.writerows(layer)
        # #pickle.dump({"state" : self.agent.states}, open("C:/Users/LeoCharlie/PycharmProjects/DeepLearning/data/" + "Me" + ".txt", "wb"))
        # print("Data successfully saved")
        # self.agent.states = []
        # print("Data flushed")
        # print("Done")

def parse_replay(replay_batch, agent_module, agent_cls):
    for replay in replay_batch:
        try:
            parser = Parser(replay, agent_cls())
            print("Got to Start")
            parser.start()
        except Exception as e:
            print("Exception was made")
            print(e)

def main(unused):
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    processes = 1#FLAGS.procs
    replay_folder = FLAGS.replays
    batch_size = 1#FLAGS.batch

    truePath = os.path.join(replay_folder, '*.SC2Replay')
    replays = glob.glob(truePath, recursive=True)

    start = FLAGS.start
    # Split replays into batches
    for i in tqdm(range(math.ceil(len(replays)/processes/batch_size))):
        procs = []
        # Skip to start pos
        x = i * processes * batch_size
        if x < start:
            continue
        # For each processor
        for p in range(processes):
            # Get num replay start
            xp1 = x + p * batch_size
            # Get num replay end
            xp2 = xp1 + batch_size
            xp2 = min(xp2, len(replays))
            # Give each processor their replays for this batch
            p = Process(target=parse_replay, args=(replays[xp1:xp2], agent_module, agent_cls))
            p.start()
            # Add process to list
            procs.append(p)
            # If at the end of replays, break
            if xp2 == len(replays):
                break
        # Does something?
        for p in procs:
            p.join()


if __name__ == "__main__":
    app.run(main)