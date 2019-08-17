#!/usr/bin/env python

from pysc2.lib import features, point, remote_controller, actions, sc_process
from absl import app, flags
from pysc2.env.environment import TimeStep, StepType
from pysc2 import run_configs
from pysc2.run_configs import lib
from s2clientprotocol import common_pb2
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol.sc2api_pb2 import RequestReplayInfo
import importlib


FLAGS = flags.FLAGS
flags.DEFINE_string("replay", "C:\Program Files (x86)\StarCraft II\Maps\Acropolis.SC2Replay", "Path to a replay file.")
flags.DEFINE_string("agent", "ObserverAgent.ObserverAgent", "Path to an agent.")
#flags.mark_flag_as_required("replay")
#flags.mark_flag_as_required("agent")

class ReplayEnv:
    screen_size_px=(84, 84)
    minimap_size_px=(84, 84)
    map_size=(153,148)
    camera_width = 300
    def __init__(self,
                 replay_file_path,
                 agent,
                 player_id=1,
                 discount=1.,
                 step_mul=1):

        
        
        self.agent = agent
        self.discount = discount
        self.step_mul = step_mul
        # lib.version_dict
        self.run_config = run_configs.get()
        # self.run_config.lib.
        self.sc2_proc = self.run_config.start()
        self.controller = self.sc2_proc.controller
        # self.sc2_proc.version = sc_pb.RequestReplayInfo.download_data
        replay_data = self.run_config.replay_data(replay_file_path)
        ping = self.controller.ping()
        # sc_process.
        self.info = self.controller.replay_info(replay_data)
        if not self._valid_replay(self.info, ping):
            raise Exception("{} is not a valid replay file!".format(replay_file_path))

        _screen_size_px = point.Point(*self.screen_size_px)
        _minimap_size_px = point.Point(*self.minimap_size_px)
        interface = sc_pb.InterfaceOptions(
            raw=False, score=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=self.camera_width,crop_to_playable_area=True))
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
        if (info.HasField("error") or
                    info.base_build != ping.base_build or  # different game version
                    info.game_duration_loops < 10 or
                    len(info.player_info) != 2):
            # Probably corrupt, or just not interesting.
            return False
#   for p in info.player_info:
#       if p.player_apm < 10 or p.player_mmr < 1000:
#           # Low APM = player just standing around.
#           # Low MMR = corrupt replay or player who is weak.
#           return False
        return True

    def start(self):
        _features = features.features_from_game_info(self.controller.game_info())

        _features.init_camera(features.Dimensions(self.screen_size_px,self.minimap_size_px), point.Point(*self.map_size), self.camera_width)
        while True:
            self.controller.step(self.step_mul)
            obs = self.controller.observe()
            try:
                agent_obs = _features.transform_obs(obs)
            except:
                pass
            
            #screenpoint = (42, 42)
            #screenpoint = point.Point(*screenpoint)
            if (len(obs.actions) == 0):
                continue
            #else:
            #    print(obs.actions)

            #if obs.observation.game_loop in config.actions:
                #func = config.actions[o.game_loop](obs)

           #_features.reverse_action(obs.actions[1])

            #action = _features.transform_action(obs.observation, actions.FUNCTIONS.move_camera([42,42]))
            #self.controller.act(action)

            #self.assertEqual(actions.FUNCTIONS.move_camera.id, func.function)

            #s2clientprotocol_dot_common__pb2._POINT2D
            #actions.FUNCTIONS.move_camera(screenpoint)
            #remote_controller.RemoteController.act(actions.move_camera(actions.FUNCTIONS.move_camera,['FEATURES'], screenpoint))
            #action_observer_camera_move = (sc_pb.ActionObserverCameraMove(world_pos = screenpoint))
            #sc_pb.RequestObserverAction
            #screenpoint.assign_to(action_observer_camera_move.world_pos)
            #self.controller.act(sc_pb.ActionObserverCameraMove(world_pos=screenpoint))
            #sc_pb.RequestObserverAction(actions=[sc_pb.ObserverAction(player_perspective=sc_pb.ActionObserverPlayerPerspective(player_id=2))])
            #obsAction = self.controller.act(sc_pb.RequestObserverAction(actions=[sc_pb.ObserverAction(player_perspective=sc_pb.ActionObserverPlayerPerspective(player_id=2))]))#   [sc_pb.ActionObserverCameraMove(distance=50)]))
            #screenpoint.assign_to(obsAction.camera_move.world_pos)
            #remote_controller.RemoteController.actions
            if obs.player_result: # Episide over.
                self._state = StepType.LAST
                discount = 0
            else:
                discount = self.discount

            #if (_features.reverse_action(obs.actions[0]).function == actions.FUNCTIONS.select_rect.id):
            agent_obs = _features.transform_obs(obs)

            #self._episode_steps += self.step_mul

            step = TimeStep(step_type=self._state, reward=0,
                            discount=discount, observation=agent_obs)
            acts = []
            for action in obs.actions:
                for num in self.agent.action_dict.keys():
                    if (format(_features.reverse_action(action).function) == num):
                        acts.append(_features.reverse_action(action))
                        break

            data = self.controller.data()
            self.agent.step(step, self.info, acts)
            #offset = self.agent.step(step, self.info)
            #print(_features.reverse_action(obs.actions[0]))
            #print ("+")
            #print(offset)
            if obs.player_result:
                break

            self._state = StepType.MID


def main(unused):
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)

    G_O_O_D_B_O_Y_E = ReplayEnv(FLAGS.replay, agent_cls())
    G_O_O_D_B_O_Y_E.start()

if __name__ == "__main__":
    app.run(main)
