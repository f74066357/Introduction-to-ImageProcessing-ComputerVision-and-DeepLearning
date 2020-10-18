import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Q4_Image/Parrot.png')

cv2.namedWindow('4', cv2.WINDOW_NORMAL)
cv2.imshow('4',img)

tx=300
ty=200
angle=30
scale=0.9
#cv2.namedWindow('move', cv2.WINDOW_NORMAL)
rows, cols = img.shape[0:2]
M = np.float32([[1,0,tx],[0,1,ty]])
dst = cv2.warpAffine(img, M, (cols,rows))
#cv2.imshow('move', dst)

#cv2.namedWindow('rotate', cv2.WINDOW_NORMAL)
rows, cols = dst.shape[0:2]
M = cv2.getRotationMatrix2D((cols/2,rows/2),angle, scale)
rotate = cv2.warpAffine(dst, M, (cols, rows))
#cv2.imshow('rotate', rotate)



cv2.waitKey(0)
cv2.destroyAllWindows()