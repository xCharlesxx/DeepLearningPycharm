import numpy as np
import math

from pysc2.lib import actions as sc_action
from pysc2.lib import static_data
from pysc2.lib import units
from s2clientprotocol.raw_pb2 import Unit
from Constants import const
import datetime
class Translator(object):
    unit_dict = dict(zip(np.insert(np.array(static_data.UNIT_TYPES), 0, 0, axis=0), ([0, 0.00423729, 0.00847458, 0.01271186, 0.01694915, 0.02118644,
                                         0.02542373, 0.02966102, 0.03389831, 0.03813559, 0.04237288, 0.04661017,
                                         0.05084746, 0.05508475, 0.05932203, 0.06355932, 0.06779661, 0.0720339,
                                         0.07627119, 0.08050847, 0.08474576, 0.08898305, 0.09322034, 0.09745763,
                                         0.10169492, 0.1059322,  0.11016949, 0.11440678, 0.11864407, 0.12288136,
                                         0.12711864, 0.13135593, 0.13559322, 0.13983051, 0.1440678,  0.14830508,
                                         0.15254237, 0.15677966, 0.16101695, 0.16525424, 0.16949153, 0.17372881,
                                         0.1779661,  0.18220339, 0.18644068, 0.19067797, 0.19491525, 0.19915254,
                                         0.20338983, 0.20762712, 0.21186441, 0.21610169, 0.22033898, 0.22457627,
                                         0.22881356, 0.23305085, 0.23728814, 0.24152542, 0.24576271, 0.25,
                                         0.25423729, 0.25847458, 0.26271186, 0.26694915, 0.27118644, 0.27542373,
                                         0.27966102, 0.28389831, 0.28813559, 0.29237288, 0.29661017, 0.30084746,
                                         0.30508475, 0.30932203, 0.31355932, 0.31779661, 0.3220339,  0.32627119,
                                         0.33050847, 0.33474576, 0.33898305, 0.34322034, 0.34745763, 0.35169492,
                                         0.3559322,  0.36016949, 0.36440678, 0.36864407, 0.37288136, 0.37711864,
                                         0.38135593, 0.38559322, 0.38983051, 0.3940678,  0.39830508, 0.40254237,
                                         0.40677966, 0.41101695, 0.41525424, 0.41949153, 0.42372881, 0.4279661,
                                         0.43220339, 0.43644068, 0.44067797, 0.44491525, 0.44915254, 0.45338983,
                                         0.45762712, 0.46186441, 0.46610169, 0.47033898, 0.47457627, 0.47881356,
                                         0.48305085, 0.48728814, 0.49152542, 0.49576271, 0.5,        0.50423729,
                                         0.50847458, 0.51271186, 0.51694915, 0.52118644, 0.52542373, 0.52966102,
                                         0.53389831, 0.53813559, 0.54237288, 0.54661017, 0.55084746, 0.55508475,
                                         0.55932203, 0.56355932, 0.56779661, 0.5720339,  0.57627119, 0.58050847,
                                         0.58474576, 0.58898305, 0.59322034, 0.59745763, 0.60169492, 0.6059322,
                                         0.61016949, 0.61440678, 0.61864407, 0.62288136, 0.62711864, 0.63135593,
                                         0.63559322, 0.63983051, 0.6440678,  0.64830508, 0.65254237, 0.65677966,
                                         0.66101695, 0.66525424, 0.66949153, 0.67372881, 0.6779661,  0.68220339,
                                         0.68644068, 0.69067797, 0.69491525, 0.69915254, 0.70338983, 0.70762712,
                                         0.71186441, 0.71610169, 0.72033898, 0.72457627, 0.72881356, 0.73305085,
                                         0.73728814, 0.74152542, 0.74576271, 0.75,       0.75423729, 0.75847458,
                                         0.76271186, 0.76694915, 0.77118644, 0.77542373, 0.77966102, 0.78389831,
                                         0.78813559, 0.79237288, 0.79661017, 0.80084746, 0.80508475, 0.80932203,
                                         0.81355932, 0.81779661, 0.8220339,  0.82627119, 0.83050847, 0.83474576,
                                         0.83898305, 0.84322034, 0.84745763, 0.85169492, 0.8559322,  0.86016949,
                                         0.86440678, 0.86864407, 0.87288136, 0.87711864, 0.88135593, 0.88559322,
                                         0.88983051, 0.8940678,  0.89830508, 0.90254237, 0.90677966, 0.91101695,
                                         0.91525424, 0.91949153, 0.92372881, 0.9279661,  0.93220339, 0.93644068,
                                         0.94067797, 0.94491525, 0.94915254, 0.95338983, 0.95762712, 0.96186441,
                                         0.96610169, 0.97033898, 0.97457627, 0.97881356, 0.98305085, 0.98728814,
                                         0.99152542, 0.99576271, 1.0])))
    def crop_feature_layers(self, stencil, featureLayers, new_width, new_height):
        # Remove unused layers
        # used = [0, 1, 2, 5, 6, 7, 9, 11, 14, 16]
        unused = [3, 4, 8, 10, 12, 13, 15]
        result = np.argmax(stencil != 0)
        stencilY = int(result / (const.WorldSize().x*4))
        stencilX = result % (const.WorldSize().x*4)
        featureLayers = np.delete(featureLayers, unused, axis=0)
        return featureLayers[0:10, stencilY:stencilY + new_height, stencilX:stencilX + new_width]

    def translate_feature_layers(self, featurelayers):
        # print("Translate")
        # print(datetime.datetime.now().time())
        # This forces unit type to be defined by their index and then between 0 and 1
        # unit_type = featurelayers[4]
        #
        unit_type_compressed = np.vectorize(self.unit_dict.__getitem__, otypes=[float])(featurelayers[4])

        # for y in range(len(featurelayers[4])):
        #     for x in range(len(featurelayers[4][y])):
        #         if unit_type[y][x] > 0 and unit_type[y][x] in static_data.UNIT_TYPES:
        #             unit_type_compressed[y][x] = static_data.UNIT_TYPES.index(unit_type[y][x]) / len(static_data.UNIT_TYPES)

        newFeatureLayers = [
             featurelayers[0] / 255,               # height_map 0 - 256
             featurelayers[1] / 2,                 # visibility 0 - 2
             featurelayers[2],                     # creep 0 - 1
             (featurelayers[3] == 1).astype(int),  # own_units 0 - 1
             (featurelayers[3] == 3).astype(int),  # neutral_units 0 - 1
             (featurelayers[3] == 4).astype(int),  # enemy_units 0 - 1
             unit_type_compressed,                 # unit types 0 - 1 << Categorical
             featurelayers[5],                     # selected 0 - 1
             featurelayers[6] / 255,               # unit_hit_points_ratio 0 - 256
             featurelayers[7] / 255,               # energy ratio 0 - 256
             featurelayers[8] / 15,                # unit_density 0 - 16
             featurelayers[9] / 15                 # effects 0 - 16 << Categorical
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