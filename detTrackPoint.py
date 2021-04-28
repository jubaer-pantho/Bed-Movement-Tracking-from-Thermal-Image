# author: Md Jubaer Hossain Pantho
# Developed for Smart Systems Lab
# University of Florida

import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

def DetectTrackPoint(dir_src, dir_dest, dist =30, y_offset=300, x_offset=100):

    img = cv2.imread(dir_src, cv2.IMREAD_UNCHANGED)
    height, width, depth = img.shape
    gray2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray2,(7,7),0)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=4)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    '''
    # Use this portion for contour detection if needed.
    # The largest region comprises the person.
    contours, hierarchy = cv2.findContours(unknown, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    np.set_printoptions(threshold=sys.maxsize)

    len1 = len(contours[0])
    len2 = len(contours[1])
    len3 = len(contours[2])

    if(len1>=len2 and len1 >=len3):
        maxContour = contours[0]
    elif(len2>=len1 and len2 >=len3):
        maxContour = contours[1]
    else:
        maxContour = contours[2]

    img = cv2.drawContours(img, maxContour, -1, (0, 255, 0), 2)
    '''

    trackPointH = []
    trackPointW = []
    for j in range(x_offset, (width-x_offset), dist):
        for i in range(y_offset, height, 1):

            if (unknown[i,j] == 0):
                trackPointH.append(i)
                trackPointW.append(j)
                break

            if (i == (height - 1)):
                trackPointH.append(0)
                trackPointW.append(j)
                break

    radius = 3
    thickness = 2
    color = (255, 0, 0)
    for i in range(len(trackPointW)):
        center_coordinates = (trackPointW[i], trackPointH[i])
        img = cv2.circle(img, center_coordinates, radius, color, thickness)

    #uncomment for debugging purposes
    #cv2.imshow("image", img)
    #cv2.waitKey(0)
    cv2.imwrite(dir_dest, img)

    return trackPointH, trackPointW

