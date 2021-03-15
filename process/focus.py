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
    def __init__(self, fname, params):
        ''' Store key parameters. Split the channels into their respectve zstacks. '''
        self.fname = fname
        self.raw = tf.imread(self.fname)
        self.channels = [[] for i in range(self.raw.shape[1])]
        self.params = params
        for c in range(0,self.raw.shape[1]):
            for zstack in self.raw[:,c,:,:]:
                self.channels[c].append((zstack/256).astype('uint8'))
                cv2.imwrite('check.png', (zstack/256).astype('uint8'))
        
    def __homography(self, zstackKeyPoints, motherKeyPoints, matched):
        ''' Finds homography between two images to warp images for alignment. '''
        zstackPoints = motherPoints = np.zeros((len(matched), 1, 2), dtype=np.float32)

        for i in range(0,len(matched)):
            zstackPoints[i] = zstackKeyPoints[matched[i].queryIdx].pt
            motherPoints[i] = motherKeyPoints[matched[i].trainIdx].pt


        homography, mask = cv2.findHomography(zstackPoints, motherPoints, cv2.RANSAC, ransacReprojThreshold=2.0)

        return homography
        
    def __align(self):
        ''' Aligns the images of each channel in the TIFF. '''
        # ORB is fast and works well for most images
        detector = cv2.ORB_create()

        # Prepare the results table
        results = [[] for channel in self.channels]
        c = 0
        for channel in self.channels:
            mother = channel[0] # Take the top image as the mother iamge to align to
            # Get the key points and descriptor using the detector
            motherKeyPoints, motherDesc = detector.detectAndCompute(mother, None)
            for zstack in range(1,len(channel)):
                zstackKeyPoints, zstackDesc = detector.detectAndCompute(channel[zstack], None)

                # If we are using ORB then we should just brute force it
                # Hamming and crosscheck to attempt some error correction
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                raw = matcher.match(zstackDesc, motherDesc)
                
                sortedRaw = sorted(raw, key=lambda match: match.distance)
                topMatches = sortedRaw[0:128] # Could bump this up, future parameter?

                # Find the homography of these two images
                homography = self.__homography(zstackKeyPoints, motherKeyPoints, topMatches)
                # Now actually do the alignment step
                alignedImage = cv2.warpPerspective(channel[zstack], homography,  (channel[zstack].shape[1], channel[zstack].shape[0]), flags=cv2.INTER_LINEAR)
                results[c].append(alignedImage)
            c += 1 # increment channel count

        return results

    def __findLaplacian(self, image):
        ''' Computes Laplacian of the image getting what's in focus (edges really). '''
        # Grab the specified parameters, this defaults to 5,5
        kernel, blur = self.params.get('lapKernel'), self.params.get('lapBlur')
        # Apply blur so we can easily calculate the laplacian
        gauss = cv2.GaussianBlur(image, (blur,blur), 0) 
        # Calculate the laplacian
        laplace = cv2.Laplacian(gauss, cv2.CV_64F, ksize=kernel) 

        return laplace

    def __findPrewitt(self, image):
        ''' Compute Prewitt's operator on the image to find what's in focus. '''
    def __doRebuild(self, channels=[]):
        ''' Rebuilds the TIFF using the focus stacked channels. '''
        print(np.asarray(channels).shape)
        tf.imwrite('resutl.tif', np.asarray(channels), photometric='minisblack')

    def getFocused(self, tiffPerChannel=False):
        ''' Focus stacks each individual channel and then rebuilds TIFF for output. '''
        aligned = self.__align()
        focused = []
        for channel in aligned:
            processed = []
            for zstack in channel:
                processed.append(self.__findLaplacian(zstack))

            stacked = np.zeros(shape=channel[0].shape, dtype=channel[0].dtype)
            
            absolute = np.absolute(np.asarray(processed))
            maxima = absolute.max(axis=0)
            boolMask = processed == maxima
            mask = boolMask.astype(np.uint8)

            for zstack in range(0,len(channel)):
                stacked = cv2.bitwise_not(channel[zstack], stacked, mask=mask[zstack])

            focused.append(255-stacked)
        
        if not tiffPerChannel:
            self.__doRebuild(focused)