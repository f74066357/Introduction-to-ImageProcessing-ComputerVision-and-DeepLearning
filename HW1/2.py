
import cv2
import numpy as np
from matplotlib import pyplot as plt


    
#gray
img = cv2.imread('Q3_Image/Chihiro.jpg', cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('gaussian', cv2.WINDOW_NORMAL)
out=Gaussian(img)
cv2.imshow('gaussian',out)

cv2.namedWindow('Sobelx', cv2.WINDOW_NORMAL)
out_sobelx = Sobelx(out)
cv2.imshow('Sobelx', out_sobelx)

cv2.namedWindow('Sobely', cv2.WINDOW_NORMAL)
out_sobely = Sobely(out)
cv2.imshow('Sobely', out_sobely)

cv2.namedWindow('Sobel', cv2.WINDOW_NORMAL)
out_sobel = Sobel(out,80)
cv2.imshow('Sobel', out_sobel)

cv2.waitKey(0)
cv2.destroyAllWindows()