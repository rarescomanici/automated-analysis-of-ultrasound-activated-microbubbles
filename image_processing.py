import numpy as np
import cv2

# returns keypoint objects (from cv2) which encode coordinates etc.
def process(image_data):

    processing_data = image_data.copy() # saving for data integrity

    threshold = 200 # threshold for binarisation, possible param for ML training
    maxval = 255 # max pixel value for grayscale

    ## Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = maxval-1

    # Filter by Color
    params.filterByColor = True
    params.blobColor = 255 # detecting white in binary mask

    # Filter by Area
    params.filterByArea = True
    params.minArea = 300 # this works with most blobs
    params.maxArea = 50000 # as big as possible without going into background

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.2

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.9

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Set up the detector with parameters
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = [] # setting up keypoint list

    # applying the blob detector on ROIs
    for image_index in range(processing_data.shape[0]):

        #mean = np.mean(processing_data[image_index]) # finding the mean to adapt to changes in brightness, contrast for now - possible ML training param for gamma
        [alpha, beta, gamma] = [1, 0.0, 0.35] # adjustments for contrast, brightness, gamma correction
        mask = cv2.medianBlur(processing_data[image_index], 3) # median filter function does not work on 3d arrays
        # the array needs to be iterated through
        mask = mask**gamma * alpha + beta # masking values with alpha, beta, gamma
        # alpha, beta params don't do anything when the data is normalized so they are set to 1,0 for now

        ## Blob detection approach
        im_in = ((mask / mask.max())*255).astype(np.uint8) # normalizing
        th, im_th = cv2.threshold(im_in, threshold, maxval, cv2.THRESH_BINARY_INV)

        # Copy the thresholded image
        im_floodfill = im_th.copy()

        # Mask used to flood filling
        # Notice the size needs to be 2 pixels than the image
        h, w = im_th.shape[:2]
        im_mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, im_mask, (0,0), 255)

        if np.mean(im_floodfill != 255): # If the floodfill did not fill the entire image (i.e. image turned white)

            # Invert floodfilled image
            im_floodfill_inv = cv2.bitwise_not(im_floodfill)

            # Combine the two images to get the foreground
            im_out = im_th | im_floodfill_inv

        else:

            im_out = im_th # If the image got compromised by the floodfill, returns the same image


        # Detect blobs
        im_keypoints = detector.detect(im_out) # keypoint stores (x,y), size, angle etc.
        # returns tuples of keypoints in the image

        # # Draw detected blobs as red circles
        # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(im_out, im_keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # # Show keypoints
        #cv2.imshow("Keypoints", im_with_keypoints)
        #cv2.waitKey(0)

        # appending to keypoint list
        keypoints.append(np.flip(np.asarray(im_keypoints, dtype="object"))) # flipping to get the uppermost Y coordinate first; OpenCV detects them the other way around

    return np.asarray(keypoints, dtype="object")
