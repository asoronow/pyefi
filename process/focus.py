import cv2
import tifffile as tf
import numpy as np

class TIFF():
    '''
    A small class for handling TIFF files. Gives usefull stats and provides
    cataloged channel data. Raw data may also be accessed through it.
    
    The main reason for having a TIFF object is the ability to create
    lists of objects to process rather than lists of arrays. Overall I think
    it makes for a cleaner workflow.
    '''
    def __init__(self, fname):
        self.fname = fname
        self.raw = tf.imread(self.fname)
        self.channels = []
        self.zheight = self.raw.shape[0]
        
        for c in range(0,self.raw.shape[1]):
            self.channels.append(self.raw[:,c,:,:])
