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

    def adaptativeSegmentation(image):
        return cv2.adaptiveThreshold(image, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 3, 0)

    def otsuSegmentation(image):
        _, segimage = cv2.threshold(image, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return segimage


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

    """
    Watershed with markers
    Given the intensity function I for a pixel
    foreground pixels: I(p) < mean(I) - 2 * stddev(I)
    background pixels: I(p) > mean(I) - stddev(I)
    """
    def watershedSegmentation(image):
        # calculate fg and bg
        fg = image < np.mean(image) - 2 * np.sqrt(np.var(image))
        bg = image > np.mean(image) - np.sqrt(np.var(image))
        fg = fg.astype("uint8")
        bg = bg.astype("uint8")

        # get connected components of fg
        numcom, marker = cv2.connectedComponents(fg)
        
        # background is considered a connected component too
        marker += bg * numcom

        # opencv's watershed only works in 3channel colored images
        colorimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        shed = cv2.watershed(colorimg, marker)
        
        # watershed now found all background
        result = shed != numcom

        # but it also set the border as -1 (border), so lets correct that
        result[0] = 0
        result[-1] = 0
        result[:, 0] = 0
        result[:, -1] = 0

        result = result.astype("uint8")
        result = result * 255
        return result
        

    methods = {'otsu': otsuSegmentation, 'adaptative': adaptativeSegmentation,
               'yen': yenSegmentation, 'watershed': watershedSegmentation}

    def getAvailableMethods():
        return list(ImageSegmenter.methods.keys())
