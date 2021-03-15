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
    ''' Gets command line parameters executes program. '''
    defaults = {
        "lapKernel":13, 
        "lapBlur":13
        }
    TIFF("m107.ome.tif", defaults).getFocused()