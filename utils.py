import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

class printFPS:
    '''
    used to calculate and print FPS
    '''
    def __init__(self, cap):
        self.modulo = 10    #print every 10th frame
        self.i = 0
        self.cap = cap

    def __call__(self):
        if(self.i%self.modulo==0):
            self.i=0
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        self.i+=1

def drawGrid(frame, height, width, ticks=4):
    '''
    draws a grid on the frame for display
    :param frame:
    :param height:
    :param width:
    :param ticks:
    :return:
    '''
    color = (155, 155, 155)
    ptsize = 1

    # first vertical
    inc = int(width / ticks)
    size = 0
    for i in range(ticks):
        cv2.line(frame, (size, 0), (size, height), color, ptsize)
        size += inc

    # then horizontal
    inc = int(height / ticks)
    size = 0
    for i in range(ticks):
        cv2.line(frame, (0, size), (width, size), color, ptsize)
        size += inc
