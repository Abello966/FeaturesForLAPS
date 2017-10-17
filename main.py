from ImageFeatureExtractor import ImageFeatureExtractor
import time
import numpy as np
import cv2

ENTRY = "LRoot_clean.npz" 
OUT = "LRoot_clean_extracted.npz"

data = np.load(ENTRY)
X = data['arr_0']
y = data['arr_1']

extractor = ImageFeatureExtractor('otsu')
before = time.clock()
print("Begin!")
for i in range(len(X)):
    X[i] = extractor.extract(X[i])
    if i % 100 == 0:
        now = time.clock()
        print("{} - {}s".format(i, now - before))
        before = now

np.savez(OUT, arr_0=X, arr_1 = y)
