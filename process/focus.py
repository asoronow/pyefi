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
        ''' Store key parameters. Split the channels into their respectve zstacks. '''
        self.fname = fname
        self.raw = tf.imread(self.fname)
        self.channels = []
        self.focusedChannels = []
        self.zheight = self.raw.shape[0]
        
        for c in range(0,self.raw.shape[1]):
            self.channels.append(self.raw[:,c,:,:])
    
    def __homography(self, image1, image2):
        ''' Finds homography between two images to warp images for alignment. '''
    def __align(self):
        ''' Aligns the images of each channel in the TIFF. '''
    def __findLaplacian(self, image):
        ''' Computes Laplacian of the image getting what's in focus (edges really). '''
    def __findPrewitt(self, image):
        ''' Compute Prewitt's operator on the image to find what's in focus. '''
    def __doRebuild(self, channels=[]):
        ''' Rebuilds the TIFF using the focus stacked channels. '''
    def getFocused(self, tiffPerChannel=False):
        ''' Focus stacks each individual channel and then rebuilds TIFF for output. '''