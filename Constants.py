from pysc2.lib import point
class const:
    def InputSize():
       return 84
    def OutputSize():
        return 2
    def ScreenSize():
       return 304#point.Point(304,294)#[153, 153]
    def MiniMapSize():
       return 304#point.Point(304,294)#[153, 153]
    def WorldSize(x_y):
        if (x_y == 'x'):
            return 153 
        return 148
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




