import numpy as np
import math

from pysc2.lib import actions as sc_action
from pysc2.lib import static_data
from pysc2.lib import units
from s2clientprotocol.raw_pb2 import Unit

class Translator(object):
    """description of class"""
    def crop_feature_layers(self, stencil, featureLayers, new_width, new_height):
        newFeature = []
        for layer in featureLayers:
            input = layer
            newInput = np.zeros((new_height,new_width),int)
            counterx = 0
            countery = 0
            for numy, y in enumerate(stencil):
                for numx, x in enumerate(y): 
                    if (x != 0):
                        newInput[countery][counterx] = input[numy][numx]
                        counterx+=1
                    if (counterx == new_width):
                        countery+=1
                        counterx=0
            newFeature.append(newInput)
        return newFeature   

    def translate_feature_layers(self, featurelayers):
        # This forces unit type to be defined by their index and then between 0 and 1
        unit_type = featurelayers[6]
        unit_type_compressed = np.zeros(featurelayers[6].shape, dtype=np.float)
        for y in range(len(featurelayers[6])):
            for x in range(len(featurelayers[6][y])):
                if unit_type[y][x] > 0 and unit_type[y][x] in static_data.UNIT_TYPES:
                    unit_type_compressed[y][x] = static_data.UNIT_TYPES.index(unit_type[y][x]) / len(static_data.UNIT_TYPES)
        units
        newFeatureLayers = [
             featurelayers[0] / 255,               # height_map 0 - 256
             featurelayers[1] / 2,                 # visibility 0 - 2
             featurelayers[2],                     # creep 0 - 1
           # featurelayers[3]                      # power 0 - 1
             (featurelayers[5] == 1).astype(int),  # own_units 0 - 1
             (featurelayers[5] == 3).astype(int),  # neutral_units 0 - 1
             (featurelayers[5] == 4).astype(int),  # enemy_units 0 - 1
             unit_type_compressed,                 # unit types 0 - 1 << Categorical
             featurelayers[7],                     # selected 0 - 1
             featurelayers[9] / 255,               # unit_hit_points_ratio 0 - 256
             featurelayers[11] / 255,              # energy ratio 0 - 256
             featurelayers[13] / 255,              # shields ratio 0 - 256
             featurelayers[14],                    # unit_density 0 - 16
             featurelayers[16]                     # effects 0 - 16 << Categorical
        ]
        return newFeatureLayers

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



#class Zerg(enum.IntEnum):
#  """Zerg units."""
#  Baneling = 9
#  BanelingBurrowed = 115
#  BanelingCocoon = 8
#  BanelingNest = 96
#  BroodLord = 114
#  BroodLordCocoon = 113
#  Broodling = 289
#  BroodlingEscort = 143
#  Changeling = 12
#  ChangelingMarine = 15
#  ChangelingMarineShield = 14
#  ChangelingZealot = 13
#  ChangelingZergling = 17
#  ChangelingZerglingWings = 16
#  Corruptor = 112
#  CreepTumor = 87
#  CreepTumorBurrowed = 137
#  CreepTumorQueen = 138
#  Drone = 104
#  DroneBurrowed = 116
#  Cocoon = 103
#  EvolutionChamber = 90
#  Extractor = 88
#  GreaterSpire = 102
#  Hatchery = 86
#  Hive = 101
#  Hydralisk = 107
#  HydraliskBurrowed = 117
#  HydraliskDen = 91
#  InfestationPit = 94
#  InfestedTerran = 7
#  InfestedTerranBurrowed = 120
#  InfestedTerranCocoon = 150
#  Infestor = 111
#  InfestorBurrowed = 127
#  Lair = 100
#  Larva = 151
#  Locust = 489
#  LocustFlying = 693
#  Lurker = 502
#  LurkerBurrowed = 503
#  LurkerDen = 504
#  LurkerCocoon = 501
#  Mutalisk = 108
#  NydusCanal = 142
#  NydusNetwork = 95
#  Overlord = 106
#  OverlordTransport = 893
#  OverlordTransportCocoon = 892
#  Overseer = 129
#  OverseerCocoon = 128
#  OverseerOversightMode = 1912
#  ParasiticBombDummy = 824
#  Queen = 126
#  QueenBurrowed = 125
#  Ravager = 688
#  RavagerBurrowed = 690
#  RavagerCocoon = 687
#  Roach = 110
#  RoachBurrowed = 118
#  RoachWarren = 97
#  SpawningPool = 89
#  SpineCrawler = 98
#  SpineCrawlerUprooted = 139
#  Spire = 92
#  SporeCrawler = 99
#  SporeCrawlerUprooted = 140
#  SwarmHost = 494
#  SwarmHostBurrowed = 493
#  Ultralisk = 109
#  UltraliskBurrowed = 131
#  UltraliskCavern = 93
#  Viper = 499
#  Zergling = 105
#  ZerglingBurrowed = 119