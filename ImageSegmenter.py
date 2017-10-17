import cv2
import numpy as np

"""
Class concentrating segmentation logic
Can be used as an object or by accessing the class methods
"""


class ImageSegmenter:

    def __init__(self, method):
        self.mode = method

    def segment(self, image):
        try:
            return self.methods[self.mode](image)
        except KeyError:
            print("KeyError in Image Segmenter: unknown method")
            raise

    def otsuSegmentation(image):
        _, segimage = cv2.threshold(image, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return segimage

    def adaptativeSegmentation(image):
        return cv2.adaptiveThreshold(image, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 3, 0)

    """
    Based off Scikit-image's implementation
    Available at:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/thresholding.py
    """
    def yenSegmentation(image):
        # simulate skimage histogram
        hist, bin_centers = np.histogram(image.ravel(), 256)
        bin_centers = np.arange(256)

        # probability mass function
        pmf = hist.astype(np.float32) / hist.sum()
        P1 = np.cumsum(pmf)
        P1_sq = np.cumsum(pmf ** 2)
        P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
        crit = np.log(((P1_sq[:-1] * P2_sq[1:]) ** -1) *
                      (P1[:-1] * (1.0 - P1[:-1])) ** 2)

        thresh = bin_centers[crit.argmax()]

        return cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY_INV)[1]

    methods = {'otsu': otsuSegmentation, 'adaptative': adaptativeSegmentation,
               'yen': yenSegmentation}

    def getAvailableMethods():
        return list(ImageSegmenter.methods.keys())
