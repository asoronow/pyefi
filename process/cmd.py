import argparse, os
from focus import TIFF
class CommandLine():
    '''
    Accept arguments and prints statuses
    '''
    def __init__(self):
        self.parser = argparse.ArgumentParser(description = 'pyefi: OME-TIFF Focus Stacking', 
                                             epilog = 'Takes a single file or directory of OME-TIFF files and focus stacks each channel.',  
                                             add_help = True, 
                                             prefix_chars = '-', 
                                             usage = '%(prog)s [options] <input file/dir> <output dir>' 
                                             )
        self.parser.add_argument('-a','--algorithm', action = 'store', choices=('laplace','sobel', 'canny') , default = 'sobel', help='The algorithm to use for focus detection. Defautl sobel.' )
        self.parser.add_argument('-lk','--lapkernel',action = 'store', type = int, default=13, help = 'Size of the kernel used for LoG, must be odd and <= 31')
        self.parser.add_argument('-sk','--sobelkernel',action = 'store', type = int, default=5, help = 'Size of the kernel used for Sobel, must be 1,3,5, or 7')
        self.parser.add_argument('-cl','--cannylo',action = 'store', type = int, default=100, help = 'Lower Canny threshold')
        self.parser.add_argument('-ch','--cannyhi',action = 'store', type = int, default=255, help = 'Upper Canny threshold')
        self.parser.add_argument('-p','--prefix',action = 'store', type = str, default= 'stacked_', help = 'Name to append to the start of output file names')
        self.parser.add_argument('input')
        self.parser.add_argument('output')

        self.args = self.parser.parse_args()

def main():
    ''' Gets command line parameters executes program. '''
    params = CommandLine().args

    if os.path.exists(params.output) and os.path.exists(params.input): # Make sure we can reach these
        if os.path.isdir(params.input): # We need to grab all the files we can process
            print("Loading files and processing, this could take a while...")
            omeTIFF = [] # List of TIFF objects to process
            for (dirpath, dirnames, filenames) in os.walk(params.input):
                for fname in filenames:
                    if 'ome.tif' in fname or 'ome.tiff' in fname: # Check the two most common extensions
                        omeTIFF.append(TIFF(os.path.join(dirpath, fname), params))
            for tiff in omeTIFF:
                tiff.getFocused()
        elif os.path.isfile(params.input): # If this is just a file then process it
            print("Loading file and processing, this could take a while...")
            TIFF(params.input, params).getFocused()

    else: # Tell them something is wrong
        print("Your input or output path was invalid!")