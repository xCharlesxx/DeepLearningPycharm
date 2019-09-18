import threading
#import win32gui
import re
class ThreadsafeButtons:
    def __init__(self):
        self.threadLock = threading.Lock()
    #def PressButtonForWindow(self,):

