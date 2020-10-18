
import cv2
import numpy as np
from matplotlib import pyplot as plt


'''    
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
'''
cv2.namedWindow('original', cv2.WINDOW_NORMAL)
img = cv2.imread('Q2_Image/Cat.png')
cv2.imshow('original', img)

cv2.namedWindow('Median filter', cv2.WINDOW_NORMAL)
# apply the 7x7 median filter on the image
median = cv2.medianBlur(img,7)
cv2.imshow('Median filter', median)

cv2.namedWindow('Gaussian Blur', cv2.WINDOW_NORMAL)
# apply the 3x3 gaussian filter on the image
gaussian = cv2.GaussianBlur(img,(3,3),0)
cv2.imshow('Gaussian Blur', gaussian)

cv2.namedWindow('Bilateral filter', cv2.WINDOW_NORMAL)
# apply the 3x3 gaussian filter on the image
bilateral = cv2.bilateralFilter(img,9,90,90)
cv2.imshow('Bilateral filter', bilateral)



cv2.waitKey(0)
cv2.destroyAllWindows()