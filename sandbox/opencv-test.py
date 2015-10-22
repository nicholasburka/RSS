#!/usr/bin/env python
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os
plt.rcParams['figure.figsize'] = (15.0, 13.0)
    
'''



COLOR THRESHOLDING





'''
def colorThresh(image):
    bgrChannels = cv2.split(image)
    blue_channel = bgrChannels[0]
    green_channel = bgrChannels[1]
    red_channel = bgrChannels[2]

    blue_thresh = 240
    red_thresh = 240
    green_thresh = 210

    max_val = 256

    #find pixels that are highly colored
    blue = cv2.inRange(blue_channel, blue_thresh, max_val)
    green = cv2.inRange(green_channel, green_thresh, max_val)
    red = cv2.inRange(red_channel, red_thresh, max_val)

    colored = cv2.bitwise_or(blue, green)
    colored = cv2.bitwise_or(colored, red)

    #find pixels that are black
    blue = cv2.inRange(blue_channel, 0, 20)
    red = cv2.inRange(red_channel, 0, 20)
    green = cv2.inRange(green_channel, 0, 20)

    black = cv2.bitwise_and(blue, red)
    black = cv2.bitwise_and(black, green)

    #find pixels that are in the floor
    floor_blue_min = 29
    floor_blue_max = 208
    floor_red_min = 25
    floor_red_max = 200
    floor_green_min = 33
    floor_green_max = 195

    blue = cv2.inRange(blue_channel, floor_blue_min, floor_blue_max)
    red = cv2.inRange(red_channel, floor_red_min, floor_red_max)
    green = cv2.inRange(green_channel, floor_green_min, floor_green_max)

    floor = cv2.bitwise_and(blue, red)
    floor = cv2.bitwise_and(floor, green)

    #combine colored and black pixels into one 2D array
    combo = cv2.bitwise_or(colored, black)

    #remove floor pixels
    combo = cv2.bitwise_and(combo, cv2.bitwise_not(floor))

    colored = cv2.bitwise_and(colored, cv2.bitwise_not(floor))

    #return colored pixels - IGNORING BLACK FOR NOW...
    return colored

def threshImage(image, thresholds):
    # image.shape[2] gives the number of channels
    # Use OpenCV to split the image up into channels, saving them in gray images
    BGRchannels = cv2.split(image)
    #print BGRchannels

    blue = BGRchannels[0]
    green = BGRchannels[1]
    red = BGRchannels[2]

    # This line creates a hue-saturation-value image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    HSVchannels = cv2.split(hsv)

    hue = HSVchannels[0]
    sat = HSVchannels[1]
    val = HSVchannels[2]

    # suppress "assigned before reference" error
    red_threshed = 1
    blue_threshed = 1
    green_threshed = 1
    hue_threshed = 1
    sat_threshed = 1
    val_threshed = 1
    threshed_image = 1

    # Threshold the image based on threshold dictionary values
    red_threshed = cv2.inRange(red, thresholds["low_red"], thresholds["high_red"], red_threshed)
    blue_threshed = cv2.inRange(blue, thresholds["low_blue"], thresholds["high_blue"], blue_threshed)
    green_threshed = cv2.inRange(green, thresholds["low_green"], thresholds["high_green"], green_threshed)
    hue_threshed = cv2.inRange(hue, thresholds["low_hue"], thresholds["high_hue"], hue_threshed)
    sat_threshed = cv2.inRange(sat, thresholds["low_sat"], thresholds["high_sat"], sat_threshed)
    val_threshed = cv2.inRange(val, thresholds["low_val"], thresholds["high_val"], val_threshed)

    # Find the different color bands


    # Multiply all the thresholded images into one "output" image, threshed_images
    threshed_image = cv2.multiply(red_threshed, green_threshed, threshed_image)
    threshed_image = cv2.multiply(threshed_image, blue_threshed, threshed_image)
    threshed_image = cv2.multiply(threshed_image, hue_threshed, threshed_image)
    threshed_image = cv2.multiply(threshed_image, sat_threshed, threshed_image)
    threshed_image = cv2.multiply(threshed_image, val_threshed, threshed_image)
    return threshed_image

# The below four functions test for "weird" objects whose colors require their own HSV profiles to be detected
#returns True if there is a white object (made of lego, hopefully) in the room
def whiteObjectTest(img, thresholds=False):
    if (not thresholds):
        thresholds =  {'low_red':0, 'high_red':255,
                           'low_green':0, 'high_green':255,
                           'low_blue':0, 'high_blue':255,
                           'low_hue':5, 'high_hue':42,
                           'low_sat':12, 'high_sat':39,
                           'low_val':205, 'high_val':255 }
    # Produce a binary image where all pixels that fall within the threshold ranges = 1, else 0
    threshed_img = threshImage(img, thresholds)

    # Get the number of pixels with a value of 1 (white pixels) 
    non_zero = cv2.countNonZero(threshed_img)

    # This number determines what "passes" and what "fails" overall
    num_pixels_required = 500

    return non_zero > num_pixels_required

def blackObjectTest(img, thresholds=False):
    if (not thresholds):
        thresholds =  {'low_red':0, 'high_red':255,
                           'low_green':0, 'high_green':255,
                           'low_blue':0, 'high_blue':255,
                           'low_hue':0, 'high_hue':133,
                           'low_sat':0, 'high_sat':177,
                           'low_val':0, 'high_val':45 }
    # Produce a binary image where all pixels that fall within the threshold ranges = 1, else 0
    threshed_img = threshImage(img, thresholds)

    # Get the number of pixels with a value of 1 (white pixels) 
    non_zero = cv2.countNonZero(threshed_img)

    # This number determines what "passes" and what "fails" overall
    num_pixels_required = 4000

    return non_zero > num_pixels_required

def baseTest(img, thresholds=False):
    if (not thresholds):
        thresholds =  {'low_red':0, 'high_red':255,
                           'low_green':0, 'high_green':255,
                           'low_blue':0, 'high_blue':255,
                           'low_hue':79, 'high_hue':114,
                           'low_sat':0, 'high_sat':45,
                           'low_val':67, 'high_val':107 }
    # Produce a binary image where all pixels that fall within the threshold ranges = 1, else 0
    threshed_img = threshImage(img, thresholds)

    # Get the number of pixels with a value of 1 (white pixels) 
    non_zero = cv2.countNonZero(threshed_img)

    # This number determines what "passes" and what "fails" overall
    num_pixels_required = 4000

    return non_zero > num_pixels_required

def boxTest(img, thresholds=False):
    if (not thresholds):
        thresholds =  {'low_red':0, 'high_red':255,
                           'low_green':0, 'high_green':255,
                           'low_blue':0, 'high_blue':255,
                           'low_hue':13, 'high_hue':15,
                           'low_sat':130, 'high_sat':189,
                           'low_val':0, 'high_val':112 }
    # Produce a binary image where all pixels that fall within the threshold ranges = 1, else 0
    threshed_img = threshImage(img, thresholds)

    # Get the number of pixels with a value of 1 (white pixels) 
    non_zero = cv2.countNonZero(threshed_img)

    # This number determines what "passes" and what "fails" overall
    num_pixels_required = 7000

    return non_zero > num_pixels_required

# Function that combines the results of a number of different color tests into a dictionary
# that states which different objects are present in an image
def coloredObjectTest(img, thresholds=False):
    # Default parameter here because defining a dictionary in the function header is messy
    if (not thresholds):
        thresholds =  {'low_red':0, 'high_red':255,
                           'low_green':0, 'high_green':255,
                           'low_blue':0, 'high_blue':255,
                           'low_hue':0, 'high_hue':255,
                           'low_sat':119, 'high_sat':255,
                           'low_val':114, 'high_val':255 }

    # Produce a binary image where all pixels that fall within the threshold ranges = 1, else 0
    threshed_img = threshImage(img, thresholds)

    # Get the number of pixels with a value of 1 (white pixels) 
    non_zero = cv2.countNonZero(threshed_img)
    print "non_zero == %d" % non_zero

    # This number determines what "passes" and what "fails" overall
    total_num_pixels_required = 2000
    # This determines how many pixels of an individual color are needed
    single_hue_num_pixels_required = 1000

    object_found = non_zero > total_num_pixels_required

    # This holds a dictionary where all of the keys are the names of colors of localisation objects
    colors_found = {'green': False,
                        'white': False, 
                        'yellow': False, 
                        'black': False,
                        'orange': False,
                        'blue': False,
                        'red': False, 
                        'base': False,
                        'box': False}


    #Tests  for objects whose colors require detection by thresholding across all of HSV space, individually
    colors_found['black'] = blackObjectTest(img)
    colors_found['white'] = whiteObjectTest(img)
    colors_found['base'] = baseTest(img)
    colors_found['box'] = boxTest(img)


    #Tests for bright colors that can be filtered out together using identical Saturation-Value thresholds
    # meaning that all of these colors are only differentiated by Hue
    if (object_found):
        # This line creates a hue-saturation-value image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # get the H from HSV
        hue_channel = cv2.split(hsv)[0]

        # Mask the H channel using the result of SV thresholding
        hue_channel = cv2.bitwise_and(threshed_img, hue_channel)

        #print "non_zero after mask == %d" % cv2.countNonZero(hue_channel)

        green_low = 36
        green_high = 86
        if cv2.countNonZero(cv2.inRange(hue_channel, green_low, green_high)) > single_hue_num_pixels_required:
            colors_found["green"] = True
        
        yellow_low = 19
        yellow_high = 23
        if cv2.countNonZero(cv2.inRange(hue_channel, yellow_low, yellow_high)) > single_hue_num_pixels_required:
            colors_found["yellow"] = True  

        blue_low = 96
        blue_high = 128
        if cv2.countNonZero(cv2.inRange(hue_channel, blue_low, blue_high)) > single_hue_num_pixels_required:
            colors_found["blue"] = True    

        orange_low = 7
        orange_high = 12
        if cv2.countNonZero(cv2.inRange(hue_channel, orange_low, orange_high)) > single_hue_num_pixels_required:
            colors_found["orange"] = True

        red_hue_num_pixels_required = 300
        red_low = 1
        red_high = 5
        if cv2.countNonZero(cv2.inRange(hue_channel, red_low, red_high)) > red_hue_num_pixels_required:
            colors_found["red"] = True

    print colors_found
    return colors_found

'''

UTILITY FUNCTIONS


'''
def displayImage(img, name, height=600.0, width=600.0):

    #resize image
    #r = height / img.shape[1]
    #dim = (int(width), int(img.shape[0] * r))
    #image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    #show image in new named window
    cv2.namedWindow(name)
    cv2.moveWindow(name, 0,0)
    cv2.imshow(name, img)
    #cv2.resizeWindow(name, int(height), int(width))

    return "True"


def colorDist(pix1, pix2):
    return math.sqrt((pix2[0] - pix1[0])**2 + (pix2[1] - pix1[1])**2 + (pix2[2] - pix1[2])**2)

minMax = {'maxR':0, 'maxG':0, 'maxB':0, 'minR':0, 'minG':0, 'minB':0}

# Display an image and allow user to click and set HSV slider values for thresholding
def analyzeImage(img):
    global minMax
    winname = "analysis"

    displayImage(img, winname)
    minMax['maxR'] = 0
    minMax['maxG'] = 0
    minMax['maxB'] = 0
    minMax['minR'] = 255
    minMax['minG'] = 255
    minMax['minB'] = 255

    def onMouse(event, x, y, flags, param):
        global maxR, maxG, maxB, minR, minG, minB     
        # clicked the left button
        if event==cv2.EVENT_LBUTTONDOWN: 
            print "x, y are", x, y, "    ",
            (b,g,r) = img[y,x]
            if b > minMax['maxB']:
                minMax['maxB'] = b
            elif b < minMax['minB']:
                minMax['minB'] = b
            if g > minMax['maxG']:
                minMax['maxG'] = g
            elif g < minMax['minG']:
                minMax['minG'] = g
            if r > minMax['maxR']:
                minMax['maxR'] = r
            elif r < minMax['minR']:
                minMax['minR'] = r
            print "r,g,b is", int(r), int(g), int(b), "    ",
            (h,s,v) = img[y,x]
            print "h,s,v is", int(h), int(s), int(v)
            down_coord = (x,y)

    cv2.setMouseCallback(winname, onMouse, None)
    
    
    thresholds =  {'low_red':0, 'high_red':255,
                       'low_green':0, 'high_green':255,
                       'low_blue':0, 'high_blue':255,
                       'low_hue':0, 'high_hue':255,
                       'low_sat':0, 'high_sat':255,
                       'low_val':0, 'high_val':255 }

    def change_slider(name, new_threshold, thresholds, img, winname):
        #change thresh
        thresholds[name] = new_threshold
        
        #get threshed image and display it
        threshed = threshImage(img, thresholds)
        displayImage(threshed, winname)

        #print the result of object test
        coloredObjectTest(img, thresholds)


    '''
    cv2.createTrackbar('low_red', winname, thresholds['low_red'], 255, 
                          lambda x: change_slider('low_red', x, thresholds, img, winname))
    cv2.createTrackbar('high_red', winname, thresholds['high_red'], 255, 
                          lambda x: change_slider('high_red', x, thresholds, img, winname))
    cv2.createTrackbar('low_green', winname, thresholds['low_green'], 255, 
                          lambda x: change_slider('low_green', x, thresholds, img, winname))
    cv2.createTrackbar('high_green', winname, thresholds['high_green'], 255, 
                          lambda x: change_slider('high_green', x, thresholds, img, winname))
    cv2.createTrackbar('low_blue', winname, thresholds['low_blue'], 255, 
                          lambda x: change_slider('low_blue', x, thresholds, img, winname))
    cv2.createTrackbar('high_blue', winname, thresholds['high_blue'], 255, 
                          lambda x: change_slider('high_blue', x, thresholds, img, winname))
    '''
    cv2.createTrackbar('low_hue', winname, thresholds['low_hue'], 255, 
                          lambda x: change_slider('low_hue', x, thresholds, img, winname))
    cv2.createTrackbar('high_hue', winname, thresholds['high_hue'], 255, 
                          lambda x: change_slider('high_hue', x, thresholds, img, winname))
    cv2.createTrackbar('low_sat', winname, thresholds['low_sat'], 255, 
                          lambda x: change_slider('low_sat', x, thresholds, img, winname))
    cv2.createTrackbar('high_sat', winname, thresholds['high_sat'], 255, 
                          lambda x: change_slider('high_sat', x, thresholds, img, winname))
    cv2.createTrackbar('low_val', winname, thresholds['low_val'], 255, 
                          lambda x: change_slider('low_val', x, thresholds, img, winname))
    cv2.createTrackbar('high_val', winname, thresholds['high_val'], 255, 
                          lambda x: change_slider('high_val', x, thresholds, img, winname))

    cv2.waitKey(0)
    print thresholds

# Display several images and allow user to click and set HSV slider values for thresholding
def analyzeImages(imgs):
    global minMax
    winname =  str(0)
    for i, img in enumerate(imgs):
        displayImage(img, str(i))

    for i, img in enumerate(imgs):
        cv2.moveWindow(str(i), i*50%1000, (i%8) * 100 % 800)
    minMax['maxR'] = 0
    minMax['maxG'] = 0
    minMax['maxB'] = 0
    minMax['minR'] = 255
    minMax['minG'] = 255
    minMax['minB'] = 255

    def onMouse(event, x, y, flags, param):
        global maxR, maxG, maxB, minR, minG, minB     
        # clicked the left button
        if event==cv2.EVENT_LBUTTONDOWN: 
            print "x, y are", x, y, "    ",
            (b,g,r) = img[y,x]
            if b > minMax['maxB']:
                minMax['maxB'] = b
            elif b < minMax['minB']:
                minMax['minB'] = b
            if g > minMax['maxG']:
                minMax['maxG'] = g
            elif g < minMax['minG']:
                minMax['minG'] = g
            if r > minMax['maxR']:
                minMax['maxR'] = r
            elif r < minMax['minR']:
                minMax['minR'] = r
            print "r,g,b is", int(r), int(g), int(b), "    ",
            (h,s,v) = img[y,x]
            print "h,s,v is", int(h), int(s), int(v)
            down_coord = (x,y)

    #cv2.setMouseCallback(winname, onMouse, None)

    thresholds =  {'low_red':0, 'high_red':255,
                       'low_green':0, 'high_green':255,
                       'low_blue':0, 'high_blue':255,
                       'low_hue':0, 'high_hue':255,
                       'low_sat':0, 'high_sat':255,
                       'low_val':0, 'high_val':255 }

    def change_slider(name, new_threshold, thresholds, imgs):
        #change thresh
        thresholds[name] = new_threshold

        for i, img in enumerate(imgs):
            #print i
            #get threshed image and display it
            print "index %d" % i
            threshed = threshImage(img, thresholds)
            displayImage(threshed, str(i))
            cv2.moveWindow(str(i), i*50%1000, (i%8) * 100 % 800)

            #print the result of object test
            coloredObjectTest(img, thresholds)

    cv2.createTrackbar('low_hue', winname, thresholds['low_hue'], 255, 
                          lambda x: change_slider('low_hue', x, thresholds, imgs))
    cv2.createTrackbar('high_hue', winname, thresholds['high_hue'], 255, 
                          lambda x: change_slider('high_hue', x, thresholds, imgs))
    cv2.createTrackbar('low_sat', winname, thresholds['low_sat'], 255, 
                          lambda x: change_slider('low_sat', x, thresholds, imgs))
    cv2.createTrackbar('high_sat', winname, thresholds['high_sat'], 255, 
                          lambda x: change_slider('high_sat', x, thresholds, imgs))
    cv2.createTrackbar('low_val', winname, thresholds['low_val'], 255, 
                          lambda x: change_slider('low_val', x, thresholds, imgs))
    cv2.createTrackbar('high_val', winname, thresholds['high_val'], 255, 
                          lambda x: change_slider('high_val', x, thresholds, imgs))

    cv2.waitKey(0)
    print thresholds
'''

SEGMENTATION

'''
# Unused, but would get the object regions from image
def getObject(threshed_img):
    img = cv2.dilate(threshed_img)


'''



CONTOUR/EDGE FINDING




'''
def contour():
    imgpath = "/afs/inf.eac.uk/user/s15/s1579555/rss/img/img_1.jpg"
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (600,600))
    #print type(img)
    #img = img.astype(np.uint8)
    img2 = img.astype(np.uint8)
    #print "\n"
    #print img

    #edge detection makes image a binary image from a colored image
    #edges = cv2.Canny(img, 100, 200)
    #get rid of uncolored pixels
    #edges = cv2.bitwise_and(edges, colorThresh(img))
    edges = colorThresh(img)

    _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    biggest = contours[0]
    for contour in contours:
        if cv2.contourArea(contour) > cv2.contourArea(biggest):
            #print contour
            biggest = contour
            print cv2.contourArea(contour) 


    #draw contours on img2, draw all of the contours in blue, with thickness 1
    cv2.drawContours(img2,contours, -1, (255, 0, 0), 1)
    cv2.drawContours(img2,[biggest], -1, (0, 255, 0), 1)



    displayImage(edges, "image")

    displayImage(img2, "image2")
    #cv2.resizeWindow("image2", 600,600)

    cv2.waitKey(0)
'''


ORB TEST



'''
def orbTest():
    res_path = "/afs/inf.eac.uk/user/s15/s1579555/rss/img/watching.png"
    resource = cv2.imread(res_path)
    scene_path = "/afs/inf.eac.uk/user/s15/s1579555/rss/img/5.jpg"
    scene = cv2.imread(scene_path)

    # Create ORB feature detector object
    orb = cv2.ORB_create(nfeatures = 1000) # Set the maximum number of returned 
                                           # feature points to 1000

    # Create a brute force matcher object.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, # Tells the matcher how to compare descriptors.
                                         # We have set it to use the hamming distance between 
                                         # descriptors as ORB provides binary descriptors.
                       crossCheck=True)  # Enable cross check. This means that the matcher
                                         # will return a matching pair [A, B] iff A is the 
                                         # closest point to B and B is the closest to A.

    # Detect the feature points in the object and the scene images
    resource_kp, resource_des = orb.detectAndCompute(resource, None)
    scene_kp, scene_des = orb.detectAndCompute(scene, None)

    # Find the matching points using brute force
    matches = bf.match(scene_des, resource_des)

    # Sort the matches in the order of the distance between the 
    # matched descriptors. Shorter distance means better match.
    matches = sorted(matches, key = lambda m: m.distance)

    # Use only the best 1/10th of matchess
    matches = matches[:len(matches)/10]

    # Visualise the detected matching pairs
    match_vis = cv2.drawMatches(scene,       # First processed image
                                scene_kp,    # Keypoints in the first image
                                resource,         # Second processed image
                                resource_kp,      # Keypoints in the second image
                                matches,     # Detected matches
                                None)        # Optional output image

    # Create numpy arrays with the feature points in order to 
    # estimate the homography between the two images.
    scene_pts = np.float32([scene_kp[m.queryIdx].pt for m in matches])
    resource_pts = np.float32([resource_kp[m.trainIdx].pt for m in matches])

    # Calculate the homography between the two images. The function works 
    # by optimising the projection error between the two sets of points.
    H, _ = cv2.findHomography(resource_pts,    # Source image
                              scene_pts,  # Destination image
                              cv2.RANSAC) # Use RANSAC because it is very likely to have wrong matches

    # Get the size of the object image
    h, w = resource.shape[:2]
    # Create an array with points at the 4 corners of the image
    bounds = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])

        
    # Project the object image corners in the scene image
    # in order to find the object
    bounds = cv2.perspectiveTransform(bounds, H).reshape(-1, 2)

    # Highlight the detected object
    for i in range(4):
        # Draw the sides of the resource by connecting consecutive points
        # The line point index is i. Get the index of the second point
        j = (i + 1) % 4
        cv2.line(match_vis,                    # Image where to draw
                 (bounds[i][0], bounds[i][1]), # First point of the line
                 (bounds[j][0], bounds[j][1]), # Second point of the line
                 (255, 255, 255),              # Colour (B, G, R)
                 3)                            # Line width in pixels

    # Draw a circle at each projected corner
    match_vis = cv2.circle(match_vis, (bounds[0][0], bounds[0][1]), 10, (255, 0, 0), -1)
    match_vis = cv2.circle(match_vis, (bounds[1][0], bounds[1][1]), 10, (0, 255, 0), -1)
    match_vis = cv2.circle(match_vis, (bounds[2][0], bounds[2][1]), 10, (0, 0, 255), -1)
    match_vis = cv2.circle(match_vis, (bounds[3][0], bounds[3][1]), 10, (0, 255, 255), -1)


    r = 1200.0 / match_vis.shape[1]
    dim = (1200, int(match_vis.shape[0] * r))
    match_vis = cv2.resize(match_vis, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("image", match_vis)

    cv2.waitKey(0)

'''



HISTOGRAM EXPERIMENTS


http://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/
'''
def getHist(img, num_bins=64):
    r_bins = cv2.calcHist([img], [2], None, [num_bins], [0, 256]) 
    g_bins = cv2.calcHist([img], [1], None, [num_bins], [0, 256])
    b_bins = cv2.calcHist([img], [0], None, [num_bins], [0, 256])
    #return RGB values in that order
    return (r_bins, g_bins, b_bins)

def getHist3D(img, num_bins=64):
    return cv2.calcHist([img], [0, 1, 2], None, [num_bins, num_bins, num_bins], [0, 256, 0, 256, 0, 256])

def calculateLocalizationHists():
    blue = cv2.imread("/afs/inf.eac.uk/user/s15/s1579555/rss/img/1.jpg")

def plotHist(hist_tuple, name):
    ylim = 1000

    plt.subplot(3, 3, 1)
    plt.title(name)

    #assume that tuple is in RGB format
    plt.plot(hist_tuple[0], 'r')
    plt.plot(hist_tuple[1], 'g')
    plt.plot(hist_tuple[2], 'b')

    plt.ylim([0, ylim])
    plt.xlim([0, 64])

def compareHist(hist_tuple1, hist_tuple2):

    #3rd parameter is an INT

    r_diff = cv2.compareHist(hist_tuple1[0], hist_tuple2[0], 1)
    g_diff = cv2.compareHist(hist_tuple1[1], hist_tuple2[1], 1)
    b_diff = cv2.compareHist(hist_tuple1[2], hist_tuple2[2], 1)

    return (r_diff, g_diff, b_diff)

def histogram():
    imgpaths = ["/afs/inf.eac.uk/user/s15/s1579555/rss/img/1.jpg", "/afs/inf.eac.uk/user/s15/s1579555/rss/img/2.jpg", "/afs/inf.eac.uk/user/s15/s1579555/rss/img/3.jpg", "/afs/inf.eac.uk/user/s15/s1579555/rss/img/4.jpg", "/afs/inf.eac.uk/user/s15/s1579555/rss/img/5.jpg"]
    scene1 = cv2.imread(imgpaths[0])
    scene2 = cv2.imread(imgpaths[1])
    scene3 = cv2.imread(imgpaths[4])

    r1, g1, b1 = getHist(scene1)
    r2, g2, b2 = getHist(scene2)
    r3, g3, b3 = getHist(scene3)

    print type(r1)

    ylim = 1000000

    plt.subplot(3, 2, 1)
    plt.title("scene 1")
    plt.plot(r1, 'r')
    plt.plot(g1, 'g')
    plt.plot(b1, 'b')
    plt.ylim([0, ylim])
    plt.xlim([0, 64])

    plt.subplot(3, 2, 2)
    plt.title("scene 2")
    plt.plot(r2, 'r')
    plt.plot(g2, 'g')
    plt.plot(b2, 'b')
    plt.ylim([0, ylim])
    plt.xlim([0, 64])

    plt.subplot(3, 2, 3)
    plt.title("scene 3")
    plt.plot(r3, 'r')
    plt.plot(g3, 'g')
    plt.plot(b3, 'b')
    plt.ylim([0, ylim])
    plt.xlim([0, 64])

    plt.show()
    cv2.waitKey(0)

def histogram2():
    scene = cv2.imread("/afs/inf.eac.uk/user/s15/s1579555/rss/img/img_2.jpg")
    diff_angle = cv2.imread("/afs/inf.eac.uk/user/s15/s1579555/rss/img/img_6.jpg")
    diff_scene = cv2.imread("/afs/inf.eac.uk/user/s15/s1579555/rss/img/img_13.jpg")

    scene_hist = getHist3D(scene)
    scene2_hist = getHist3D(diff_angle)
    scene3_hist = getHist3D(diff_scene)
    for i in range(6):
        print cv2.compareHist(scene_hist, scene_hist, i)
        print cv2.compareHist(scene_hist, scene2_hist, i)
        print cv2.compareHist(scene_hist, scene3_hist, i)
        print "TEST NUM %d" % i

    print "end test seq 1"

    scene = cv2.imread("/afs/inf.eac.uk/user/s15/s1579555/rss/img/img_17.jpg")
    diff_angle = cv2.imread("/afs/inf.eac.uk/user/s15/s1579555/rss/img/img_18.jpg")
    diff_scene = cv2.imread("/afs/inf.eac.uk/user/s15/s1579555/rss/img/img_13.jpg")

    scene_hist = getHist3D(scene)
    scene2_hist = getHist3D(diff_angle)
    scene3_hist = getHist3D(diff_scene)
    print cv2.compareHist(scene_hist, scene_hist, 1)
    print cv2.compareHist(scene_hist, scene2_hist, 1)
    print cv2.compareHist(scene_hist, scene3_hist, 1)

    print cv2.compareHist(scene_hist, scene_hist, 0)
    print cv2.compareHist(scene_hist, scene2_hist, 0)
    print cv2.compareHist(scene_hist, scene3_hist, 0)

    plotHist(scene_hist, "scene")

    #plt.show()
'''


TESTS



'''
def assertImages(imgs, color_string):
    print "\n"
    print "Testing %s image group" % color_string
    for i, x in enumerate(imgs):
        print "Index %d" % i
        test_dict = coloredObjectTest(x)
        assert(test_dict[color_string])
        test_dict[color_string] = False
        assert(not (True in test_dict.values()))

#base image path
imgpath = "/afs/inf.ed.ac.uk/user/s15/s1579555/rss/img/"

#images with colored object
img = cv2.imread(imgpath + "img_1.jpg")
img2 = cv2.imread(imgpath + "img_2.jpg")
img3 = cv2.imread(imgpath + "img_3.jpg")
img4 = cv2.imread(imgpath + "img_03.jpg")
img5 = cv2.imread(imgpath + "img_5.jpg")
img6 = cv2.imread(imgpath + "img_4.jpg")
img7 = cv2.imread(imgpath + "img_05.jpg")
img8 = cv2.imread(imgpath + "img_7.jpg")
img9 = cv2.imread(imgpath + "img_07.jpg")
img10 = cv2.imread(imgpath + "img_8.jpg")
img11 = cv2.imread(imgpath + "img_08.jpg")
img12 = cv2.imread(imgpath + "img_9.jpg")
img13 = cv2.imread(imgpath + "img_10.jpg")
img14 = cv2.imread(imgpath + "img_8.jpg")
img15 = cv2.imread(imgpath + "img_8.jpg")
img20 = cv2.imread(imgpath + "img_012.jpg")
img19 = cv2.imread(imgpath + "img_39.jpg")

#images without colored object
img16 = cv2.imread(imgpath + "img_25.jpg")
#img17 = cv2.imread(imgpath + "img_42.jpg")
img18 = cv2.imread(imgpath + "img_41.jpg")
img21 = cv2.imread(imgpath + "img_26.jpg")

images_with_color = [img, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12, img13]
images_without_color = [img16, img18, img21]

#localisation cut-out images (just colored objects, no background)
colored_objects = []
#blank wide image for bigger sliders
colored_objects.append(cv2.imread(imgpath + "wide.png"))
#colored_objects.append(cv2.imread(imgpath + "img_30.jpg"))
#colored_objects.append(cv2.imread(imgpath + "img_34.jpg"))
#colored_objects.append(cv2.imread(imgpath + "img_35.jpg"))
imgpath = "/afs/inf.ed.ac.uk/user/s15/s1579555/rss/img/localization"
list_of_paths = os.listdir(imgpath)
for x in list_of_paths:
    colored_objects.append(cv2.imread(imgpath + "/" + x))

black_images = []
black_images.append(cv2.imread(imgpath + "/" + "black.png"))
black_images.append(cv2.imread(imgpath + "/" + "black_2.png"))
black_images.append(cv2.imread(imgpath + "/" + "black_3.png"))
black_images.append(cv2.imread(imgpath + "/" + "black_2.png"))
imgpath = "/afs/inf.ed.ac.uk/user/s15/s1579555/rss/img/"
black_images.append(cv2.imread(imgpath + "/" + "img_011.jpg"))
black_images.append(cv2.imread(imgpath + "/" + "img_39.jpg"))
#analyzeImages(black_images)

'''
print colored_objects[0].shape
for i, x in enumerate(colored_objects):
    print "index %d" % i
    coloredObjectTest(x)
analyzeImages(colored_objects)
'''
#tests
assert(img.any())


imgpath = "/afs/inf.ed.ac.uk/user/s15/s1579555/rss/img/localization/"


red_images = []
red_images.append(cv2.imread(imgpath + "red_1.png"))
red_images.append(cv2.imread(imgpath + "red_2.png"))
assertImages(red_images, "red")

blue_images = []
blue_images.append(cv2.imread(imgpath + "blue_1.png"))
blue_images.append(cv2.imread(imgpath + "blue_2.png"))
blue_images.append(cv2.imread(imgpath + "blue_3.png"))
assertImages(blue_images, "blue")

green_images = []
green_images.append(cv2.imread(imgpath + "green.png"))
green_images.append(cv2.imread(imgpath + "green_2.png"))
green_images.append(cv2.imread(imgpath + "green_3.png"))
green_images.append(cv2.imread(imgpath + "light_green.png"))
green_images.append(cv2.imread(imgpath + "light_green_2.png"))
assertImages(green_images, "green")

yellow_images = []
yellow_images.append(cv2.imread(imgpath + "yellow_1.png"))
yellow_images.append(cv2.imread(imgpath + "yellow_2.png"))
yellow_images.append(cv2.imread(imgpath + "yellow_3.png"))
assertImages(yellow_images, "yellow")

orange_images = []
orange_images.append(cv2.imread(imgpath + "orange.png"))
orange_images.append(cv2.imread(imgpath + "orange_2.png"))
orange_images.append(cv2.imread(imgpath + "orange_3.png"))
assertImages(orange_images, "orange")

black_images = []
black_images.append(cv2.imread(imgpath + "black.png"))
black_images.append(cv2.imread(imgpath + "black_2.png"))
black_images.append(cv2.imread(imgpath + "black_3.png"))
assertImages(black_images, "black")

white_images = []
white_images.append(cv2.imread(imgpath + "white.png"))
white_images.append(cv2.imread(imgpath + "white_2.png"))
white_images.append(cv2.imread(imgpath + "white_3.png"))
assertImages(white_images, "white")

base_images = []
base_images.append(cv2.imread(imgpath + "base.png"))
base_images.append(cv2.imread(imgpath + "base_2.png"))
base_images.append(img20)
base_images.append(img19)
assertImages(base_images, "base")

box_images = []
box_images.append(cv2.imread(imgpath + "box.png"))
box_images.append(cv2.imread(imgpath + "box_2.png"))
assertImages(box_images, "box")
#analyzeImages(box_images + images_without_color)
#analyzeImages(images_without_color)

print "\n"
print "Testing True cases for coloredObjectTest()"
for i, x in enumerate(images_with_color):
    #analyzeImage(x)
    print i
    assert(True in coloredObjectTest(x).values())
#analyzeImages(images_with_color)

print "\n"
print "Testing False cases for coloredObjectTest()"
for i, x in enumerate(images_without_color):
    #analyzeImage(x)
    print i
    assert(not (True in coloredObjectTest(x).values()))

print "All tests passed!"
cv2.destroyAllWindows()