import numpy as np
import cv2 as cv
img = cv.imread("C:\Users\goofster\Desktop\beach-cleaning-object-detection\beach-cleaning-object-detection\JPG Compression\Image Input\castscale.jpg")

cv.imshow("Display window", img)
k = cv.waitKey(0) # Wait for a keystroke in the window
