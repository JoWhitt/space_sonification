#space_sonification/main.py
#k-means clustering: https://docs.opencv.org/master/d1/d5c/tutorial_py_kmeans_opencv.html

import cv2 as cv
import numpy as np

# read and format image
img = cv.imread('./images/PIA17811_small.jpg')
assert img is not None
cv.imshow("image",img)

Z = img.reshape((-1,3))
Z = np.float32(Z)

# apply kmeans -- to quantify image colours
# define criteria, number of clusters(K)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv.imshow('quantised image',res2)
cv.waitKey(0)
cv.destroyAllWindows()