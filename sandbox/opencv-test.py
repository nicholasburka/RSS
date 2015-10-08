#!/usr/bin/env python
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15.0, 13.0)

'''



CONTOUR/EDGE FINDING




'''
imgpath = "/afs/inf.ed.ac.uk/user/s15/s1579555/rss/img/1.jpg"
img = cv2.imread(imgpath)
img = cv2.resize(img, (600,600))
#print type(img)
#img = img.astype(np.uint8)
img2 = img.astype(np.uint8)
#print "\n"
#print img

#edge detection makes image a binary image from a colored image
edges = cv2.Canny(img, 100, 200)

_, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
'''
biggest = contours[0]
for contour in contours:
    print contour
    if cv2.contourArea(contour) > cv2.contourArea(biggest):
        print contour
        biggest = contour
        print cv2.contourArea(contour) 
'''

#draw contours on img2, draw all of the contours in blue, with thickness 1
cv2.drawContours(img2,contours, -1, (255, 0, 0), 1)



cv2.namedWindow("image")
cv2.moveWindow("image", 0,0)
cv2.imshow("image", edges)
cv2.resizeWindow("image", 600, 600)

cv2.namedWindow("image2")
cv2.moveWindow("image2", 600,600)
cv2.imshow("image2",img2)
cv2.resizeWindow("image2", 600,600)

cv2.waitKey(0)
'''


ORB TEST



'''

res_path = "/afs/inf.ed.ac.uk/user/s15/s1579555/rss/img/watching.png"
resource = cv2.imread(res_path)
scene_path = "/afs/inf.ed.ac.uk/user/s15/s1579555/rss/img/5.jpg"
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



'''
imgpaths = ["/afs/inf.ed.ac.uk/user/s15/s1579555/rss/img/1.jpg", "/afs/inf.ed.ac.uk/user/s15/s1579555/rss/img/2.jpg", "/afs/inf.ed.ac.uk/user/s15/s1579555/rss/img/3.jpg", "/afs/inf.ed.ac.uk/user/s15/s1579555/rss/img/4.jpg", "/afs/inf.ed.ac.uk/user/s15/s1579555/rss/img/5.jpg"]
scene1 = cv2.imread(imgpaths[0])
scene2 = cv2.imread(imgpaths[1])
scene3 = cv2.imread(imgpaths[4])

def getHist(img, num_bins=64):
    r_bins = cv2.calcHist([img], [2], None, [num_bins], [0, 256]) 
    g_bins = cv2.calcHist([img], [1], None, [num_bins], [0, 256])
    b_bins = cv2.calcHist([img], [0], None, [num_bins], [0, 256])
    return (r_bins, g_bins, b_bins)

r1, g1, b1 = getHist(scene1)
r2, g2, b2 = getHist(scene2)
r3, g3, b3 = getHist(scene3)

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

'''



COLOR THRESHOLDING





'''



cv2.destroyAllWindows()
