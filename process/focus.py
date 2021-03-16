import cv2, os
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
    def __init__(self, fname, params):
        ''' Store key parameters. Split the channels into their respectve zstacks. '''
        self.fname = fname
        self.outfile = fname.replace(params.input+"/", params.prefix) # Apply the prefix for output
        self.params = params # Command line arguments
        with tf.TiffFile(self.fname) as tif:
            axes = tif.series[0].axes
            data = tif.asarray()
            self.channels = [[] for c in range(0,data.shape[axes.find('C')])]
            for c in range(0,data.shape[axes.find('C')]): # Read channels into list
                if axes == 'CZYX':
                    for zstack in data[c,:,:,:]:
                        if zstack.dtype == 'uint16': # We can only handle 8-bit values with our detector
                            self.channels[c].append((zstack/256).astype('uint8')) # Convert to 8-bit if necessary
                        elif zstack.dtype == 'uint8':
                            self.channels[c].append(zstack)
                elif axes == 'ZCYX':
                    for zstack in data[:,c,:,:]:
                        if zstack.dtype == 'uint16': # We can only handle 8-bit values with our detector
                            print
                            self.channels[c].append((zstack/255).astype('uint8')) # Convert to 8-bit if necessary
                        elif zstack.dtype == 'uint8':
                            self.channels[c].append(zstack)

    def __homography(self, zstackKeyPoints, motherKeyPoints, matched):
        ''' Finds homography between two images to warp images for alignment. '''
        # Empty arrays to fill in with matched points
        zstackPoints = motherPoints = np.zeros((len(matched), 1, 2), dtype=np.float32)

        for i in range(0,len(matched)): # We train the detector with the kp of the mother image against the zstack
            zstackPoints[i] = zstackKeyPoints[matched[i].queryIdx].pt
            motherPoints[i] = motherKeyPoints[matched[i].trainIdx].pt

        # Using RANSAC to approximate the locations of absolute homography between images
        homography, mask = cv2.findHomography(zstackPoints, motherPoints, cv2.RANSAC, ransacReprojThreshold=2.0)

        return homography
        
    def __align(self):
        ''' Aligns the images of each channel in the TIFF using ORB (Oriented FAST and Rotated BRIEF). '''
        # ORB is fast and works well for most images (also not patented)
        detector = cv2.ORB_create(1000) # 1000 is the number of features to retain, higher accuracy

        # Prepare the results list
        results = [[] for channel in self.channels]
        c = 0 # Channel counter
        for channel in self.channels:
            mother = channel[0] # Take the top image as the mother image to align to
            # Get the key points and descriptor using the detector
            motherKeyPoints, motherDesc = detector.detectAndCompute(mother, None)
            for zstack in range(1,len(channel)):
                zstackKeyPoints, zstackDesc = detector.detectAndCompute(channel[zstack], None)
                # If we are using ORB then we should just brute force it
                # Hamming distance and crosscheck for feature matching
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                raw = matcher.match(zstackDesc, motherDesc)
                sortedRaw = sorted(raw, key=lambda match: match.distance)
                topMatches = sortedRaw[0:128]
                # Find the homography of these two images
                homography = self.__homography(zstackKeyPoints, motherKeyPoints, topMatches)
                # Now actually do the alignment step
                alignedImage = cv2.warpPerspective(channel[zstack], homography, (channel[zstack].shape[1], channel[zstack].shape[0]), flags=cv2.INTER_LINEAR)
                results[c].append(alignedImage)
            c += 1 # increment channel count

        return results

    def __findLoG(self, image):
        ''' Computes Laplacian of the Gaussian of the image getting what's in focus. '''
        # Grab the specified parameters, this defaults to 5,5
        kernel = blur = self.params.lapkernel
        # Apply blur so we can easily calculate the laplacian
        gauss = cv2.GaussianBlur(image, (blur,blur), 0) 
        # Calculate the laplacian
        laplace = cv2.Laplacian(gauss, cv2.CV_64F, ksize=kernel) 

        return laplace

    def __findCanny(self, image):
        ''' Computes Canny edge detection on the image to find in focus. '''
        return cv2.Canny(image, self.params.cannylo, self.params.cannyhi)

    def __findSobel(self, image):
        ''' Compute Sobel gradient of the image to find what's in focus. '''
        kernel = blur = self.params.sobelkernel # Kernel and blur for sobel

        gauss = cv2.GaussianBlur(image, (blur,blur), 0) # Apply blur

        sobelx = cv2.Sobel(gauss, cv2.CV_8U, 1, 0, ksize=kernel) # Sobel x calculation
        sobely = cv2.Sobel(gauss, cv2.CV_8U, 0, 1, ksize=kernel) # Sobel y calculation

        return sobelx + sobely # Sum the two to get the complete gradient

    def __doRebuild(self, channels=[]):
        ''' Rebuilds the TIFF using the focus stacked channels. '''
        tf.imwrite(
            os.path.join(self.params.output, self.outfile), # Output file path
            np.asarray(channels), # The newly stacked channels
            photometric='minisblack' # Tell the module these are grayscale, RGB gets funky
        )

        print(f"Finished processing {self.fname}")

    def getFocused(self):
        ''' Focus stacks each individual channel and then rebuilds TIFF for output. '''
        aligned = self.__align() # The aligned channels
        # Helpful dictionary to select the algo
        algos = {'canny':self.__findCanny,'laplace':self.__findLoG,'sobel':self.__findSobel}
        # List of the final focused channels
        focused = []

        print(f"Starting processing for {self.fname}")

        for channel in aligned:
            processed = []
            for zstack in channel:
                processed.append(algos[self.params.algorithm](zstack)) # Use the algo of choice to find in focus

            output = np.zeros(shape=channel[0].shape, dtype=channel[0].dtype) # Blank array in the shape of final image
            
            absolute = np.absolute(np.asarray(processed)) # Get the absolute values of the mask
            boolMask = absolute == absolute.max(axis=0) # We only want the maxima, making it boolean to apply bitwise
            mask = boolMask.astype(np.uint8) # convert here

            for zstack in range(0,len(channel)):
                output = cv2.bitwise_not(channel[zstack], output, mask=mask[zstack]) # Apply our newly found mask

            focused.append(255-output) # These values are flipped we want black as the minimum value

        self.__doRebuild(focused) # Recreate the new TIFF file