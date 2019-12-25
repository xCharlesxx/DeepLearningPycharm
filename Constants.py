from pysc2.lib import point
class const:
    def InputSize():
       return 84
    def OutputSize():
        return 2
    def ScreenSize():
        return point.Point(352, 352)#point.Point(352, 352)
    def MiniMapSize():
        return point.Point(352, 352)
    def WorldSize():
        return point.Point(176, 176) # 176
    def IsMicroUnit(x):
        # Empty
        if (len(x) == 0):
            return False
        if (x[0][0] == 0):
            return False
        # If any of the units selected are: Buildings, drones or larva
        for unit in x:
            if unit[0] in [104, 103, 90, 88, 102, 86, 101, 91, 94, 100, 151, 504, 97, 89, 92, 93]:
                return False
        return True
    def NumLayers():
        return 12




