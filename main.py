# author: Md Jubaer Hossain Pantho
# Developed for Smart Systems Lab
# University of Florida

import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import detTrackPoint as detTrack


def MovementDetectionAnalysis(dir_src1, dir_src2, dir_dest):

    img1 = cv2.imread(dir_src1, cv2.IMREAD_UNCHANGED)
    height1, width1, depth1 = img1.shape

    img2 = cv2.imread(dir_src2, cv2.IMREAD_UNCHANGED)
    height2, width2, depth2 = img2.shape

    if (height1 != height2 or width1 != width2 or height1 != height2):
        print("Image size mismatch. Exiting...")
        exit(1)

    trackPointH1, trackPointW1 = detTrack.DetectTrackPoint(dir_src1, 'track_points1.jpg')
    trackPointH2, trackPointW2 = detTrack.DetectTrackPoint(dir_src2, 'track_points2.jpg')


    diff = []

    newPoint = 0

    for i in range(len(trackPointH1)):
        if(trackPointH1[i] == 0 or trackPointH2[i] == 0):
            newPoint = newPoint + 1
            continue

        diff.append(abs(trackPointH2[i] - trackPointH1[i]))

        start_point = (trackPointW1[i], trackPointH1[i])
        end_point = (trackPointW2[i], trackPointH2[i])
        color = (0, 255, 0)
        thickness = 7
        img2 = cv2.line(img2, start_point, end_point, color, thickness)
        cv2.imwrite(dir_dest, img2)
      

    return diff, newPoint



diff, miss_match = MovementDetectionAnalysis('images/FLIR_21.jpeg', 'images/FLIR_22.jpeg', 'result.jpg')

mean = sum(diff) / len(diff)
variance = sum((i - mean) ** 2 for i in diff) / len(diff)
maxVal = max(diff)
minVal = min(diff)

print("accumulated change : ", sum(diff))
print("mean value : ", mean)
print("max value : ", maxVal)
print("min value : ", minVal)
print("variance of the list : ", variance)
print("miss matching points : ", miss_match)

plt.hist(diff)
plt.show()

