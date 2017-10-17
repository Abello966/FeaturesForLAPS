import cv2
import numpy as np
from main import *

test0 = cv2.imread("../qtTest_shape/p3_frame_289_213_1306.png", 0)
test1 = cv2.imread("../p015_frame_792_1503_141.bmp", 0)
extractor = ImageFeatureExtractor('otsu')

features0 = extractor.extract(test0, option='dict')
features1 = extractor.extract(test1)
features2 = extractor.extract(test0, option='dict')

for key in features0.keys():
    if features0[key] != features2[key]:
        print(key + " is different!")
else:
    print("OK")
print(features0 == features2)
