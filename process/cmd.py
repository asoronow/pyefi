import argparse

class CommandLine():
    '''
    Accept arguments and prints statuses
    '''
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # The required positional arg
        self.parser.add_argument("input", help="The file or directory you wish to process (OME-TIFF format only!)")

        optionals = {}