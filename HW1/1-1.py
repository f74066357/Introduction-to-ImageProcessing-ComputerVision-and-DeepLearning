# coding=utf-8
import cv2
import numpy as np

#1-1
'''
img = cv2.imread("Q1_image/Uncle_Roger.jpg")
width, height = img.shape[0:2]
print (width,height)
cv2.imshow("img", img)
cv2.waitKey(0)
'''
#1-2
'''
#red
imgr = cv2.imread("Q1_image/Flower.jpg")
imgr[:,:,0] = 0
imgr[:,:,1] = 0
cv2.imshow('red_img', imgr)
cv2.waitKey(0)

#blue
imgb = cv2.imread("Q1_image/Flower.jpg")
imgb[:,:,1] = 0
imgb[:,:,2] = 0
cv2.imshow('blue_img', imgb)
cv2.waitKey(0)

#green
imgg = cv2.imread("Q1_image/Flower.jpg")
imgg[:,:,0] = 0
imgg[:,:,2] = 0
cv2.imshow('green_img', imgg)
cv2.waitKey(0)
'''
#1-3
'''
cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
img = cv2.imread("Q1_image/Uncle_Roger.jpg")
cv2.imshow('original image', img)

cv2.namedWindow('result', cv2.WINDOW_NORMAL)
img2 = cv2.flip(img, 1)
cv2.imshow('result', img2)

cv2.waitKey(0)
'''
#1-4

alpha = 0.5
beta = (1.0 - alpha)
def Newshow():
    global alpha,beta
    blendimg = cv2.addWeighted(img, alpha, img2, beta, 0.0)
    cv2.imshow('blending result', blendimg)
    #print(alpha,beta)

    cv2.setTrackbarPos('blending','original image',int(alpha*255))
    cv2.setTrackbarPos('blending','result',int(beta*255))
    #cv2.setTrackbarPos('blending','blending result',int(r*255))
def Blend1(x):
    global alpha,beta
    # get current positions of trackbars
    val1=cv2.getTrackbarPos('blending','original image')
    #print(val1)
    alpha = val1/255
    beta = (1.0 - alpha)
    Newshow()
def Blend2(x):
    global alpha,beta
    val2=cv2.getTrackbarPos('blending','result')
    #print(val2)
    beta = val2/255
    alpha = (1.0 - beta)
    Newshow()
def Blend3(x):
    global alpha,beta
    valtotal = cv2.getTrackbarPos('blending','blending result')
    #print(valtotal)
    alpha=valtotal/255
    beta = (1.0 - alpha)
    Newshow()

cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
img = cv2.imread("Q1_image/Uncle_Roger.jpg")
cv2.createTrackbar('blending','original image',0,255,Blend1)
cv2.imshow('original image', img)

cv2.namedWindow('result', cv2.WINDOW_NORMAL)
img2 = cv2.flip(img, 1)
cv2.createTrackbar('blending','result',0,255,Blend2)
cv2.setTrackbarPos('blending','result',255)
cv2.imshow('result', img2)

cv2.namedWindow('blending result', cv2.WINDOW_NORMAL)
cv2.createTrackbar('blending','blending result',0,255,Blend3)
blendimg = cv2.addWeighted(img, alpha, img2, beta, 0.0)
cv2.imshow('blending result', blendimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

