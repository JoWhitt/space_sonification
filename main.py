#space_sonification/main.py

import cv2 as cv
import numpy as np


if __name__ == '__main__':
	filename = './images/PIA17811_small.jpg'
	
	img = cv.imread(filename)
	assert img is not None

	cv.imshow("space image",img)
	cv.waitKey(0)

