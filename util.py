import cv2
import numpy as np

"""
Returns main contour of binary image
"""


def getMainContour(segimage, contours, hierarchy):
    """
    get "ctr", the center of the image
    """
    ctrx = 0.5 * segimage.shape[0]
    ctry = 0.5 * segimage.shape[1]

    """
    get main contour, the one that maximizes this metric
    """
    th = 0
    for i in range(len(contours)):
        if (hierarchy[0][i][3] == -1):
            maxLength = cv2.arcLength(contours[i], True)
            mu = cv2.moments(contours[i], False)
            try:
                px = mu['m10'] / mu['m00']
                py = mu['m01'] / mu['m00']
                dis = (px - ctrx) ** 2 + (py - ctry) ** 2
                dis = np.sqrt(dis)
                tmp = maxLength / dis
            except ZeroDivisionError:
                tmp = 0

            if tmp > th:
                th = tmp
                idxMainParticle = i

    return idxMainParticle


""" Simple distance between two points """


def calcRelativeDistance(p, q):
    return np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
