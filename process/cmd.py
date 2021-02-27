import argparse
from focus import TIFF
class CommandLine():
    '''
    Accept arguments and prints statuses
    '''
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # The required positional arg
        self.parser.add_argument("input", help="The file or directory you wish to process (OME-TIFF format only!)")

        optionals = {}

def main():
    '''The main cmd handler'''
    myTiff = TIFF("process/m107t.ome.tif")
    print(myTiff.channels[0])
    print(myTiff.raw.shape)