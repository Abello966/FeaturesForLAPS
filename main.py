from ImageFeatureExtractor import ImageFeatureExtractor
import time
import numpy as np
import cv2
import argparse

# parse command line args
parser = argparse.ArgumentParser(description="Extract features from a set of images")
parser.add_argument('input',  metavar='in', type=str, nargs=1,
                    help="input grayscale numpy images")
parser.add_argument('output', metavar='out', type=str, nargs=1,
                    help="output features")
parser.add_argument('method', metavar='method', type=str, nargs=1,
                    help="segmentation method")

args = parser.parse_args()
incoming = args.input[0]
output = args.output[0]
method = args.method[0]

print(incoming)

# opens data and gets to business
data = np.load(incoming)
X = data['arr_0']
y = data['arr_1']

extractor = ImageFeatureExtractor(method)
before = time.clock()
total = 0
X = extractor.extract(X)
now = time.clock()
after = now - before

print("Done after {}s".format(after))
print("Write output")
np.savez(output, arr_0=X, arr_1 = y)
