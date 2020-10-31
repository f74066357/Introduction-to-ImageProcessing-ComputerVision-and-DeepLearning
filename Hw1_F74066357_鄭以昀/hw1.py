# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 10:47:42 2020

@author: yes19
"""
import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class Mainwindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "F7406637_HW1"
        self.left = 50
        self.top = 50
        self.width = 1000
        self.height = 360
        self.initUI()
    
    def initUI(self):
        
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        label1 = QLabel(self)
        label1.setText("1. Image Processing") 
        label1.move(50, 30)
        button1 = QPushButton("1.1 Load Image File", self)
        button1.move(50, 80) #座標
        button1.clicked.connect(self.on_click1)
        button2 = QPushButton("1.2 Color Separation", self)
        button2.move(50, 130) #座標
        button2.clicked.connect(self.on_click2)
        button3 = QPushButton("1.3 Image Flipping", self)
        button3.move(50, 180) #座標
        button3.clicked.connect(self.on_click3)
        button4 = QPushButton("1.4 Blending", self)
        button4.move(50, 230) #座標
        button4.clicked.connect(self.on_click4)

        label2 = QLabel(self)
        label2.setText("2. Image Smoothing") 
        label2.move(200, 30)
        button5 = QPushButton("2.1 Median filter", self)
        button5.move(200, 80) #座標
        button5.clicked.connect(self.on_click5)
        button6 = QPushButton("2.2 Gaussian blur", self)
        button6.move(200, 130) #座標
        button6.clicked.connect(self.on_click6)
        button7 = QPushButton("2.3 Bilateral filter", self)
        button7.move(200, 180) #座標
        button7.clicked.connect(self.on_click7)

        label3 = QLabel(self)
        label3.setText("3. Edge Detection") 
        label3.move(350, 30)
        button8 = QPushButton("3.1 Gaussian Blur", self)
        button8.move(350, 80) #座標
        button8.clicked.connect(self.on_click8)
        button9 = QPushButton("3.2 Sobel X", self)
        button9.move(350, 130) #座標
        button9.clicked.connect(self.on_click9)
        button10 = QPushButton("3.3 Sobel Y", self)
        button10.move(350, 180) #座標
        button10.clicked.connect(self.on_click10)
        button11 = QPushButton("3.4 Magnitude", self)
        button11.move(350, 230) #座標
        button11.clicked.connect(self.on_click11)

        label4 = QLabel(self)
        label4.setText("4. Transformation") 
        label4.move(500, 30)
        label5 = QLabel(self)
        label5.setText("Rotation:") 
        label5.move(500, 80)
        self.textbox1 = QLineEdit(self)
        self.textbox1.resize(80, 20)
        self.textbox1.move(560, 80)
        label6 = QLabel(self)
        label6.setText("deg") 
        label6.move(650, 80)
        label7 = QLabel(self)
        label7.setText("Scaling:") 
        label7.move(500, 130)
        self.textbox2 = QLineEdit(self)
        self.textbox2.resize(80, 20)
        self.textbox2.move(560, 130)
        label8 = QLabel(self)
        label8.setText("Tx:") 
        label8.move(500, 180)
        self.textbox3 = QLineEdit(self)
        self.textbox3.resize(80, 20)
        self.textbox3.move(560, 180)
        label9 = QLabel(self)
        label9.setText("pixel") 
        label9.move(650, 180)
        label10 = QLabel(self)
        label10.setText("Ty:") 
        label10.move(500, 230)
        self.textbox4 = QLineEdit(self)
        self.textbox4.resize(80, 20)
        self.textbox4.move(560, 230)
        label11 = QLabel(self)
        label11.setText("pixel") 
        label11.move(650, 230)
        button12 = QPushButton("4 Transformation", self)
        button12.move(500, 280) #座標
        button12.clicked.connect(self.on_click12)

        label12 = QLabel(self)
        label12.setText("5 VGG16") 
        label12.move(750, 30)
        button13 = QPushButton("1. show train images", self)
        button13.move(750, 80) #座標
        button13.clicked.connect(self.on_click13)
        button14 = QPushButton("2. show hyperparameters", self)
        button14.move(750, 130) #座標
        button14.clicked.connect(self.on_click14)
        button15 = QPushButton("3. show model structure", self)
        button15.move(750, 180) #座標
        button15.clicked.connect(self.on_click15)
        button16 = QPushButton("4. show accuracy", self)
        button16.move(750, 230) #座標
        button16.clicked.connect(self.on_click16)
        label17 = QLabel(self)
        label17.setText("5. test image index:") 
        label17.move(750, 280)
        self.textbox5 = QLineEdit(self)
        self.textbox5.resize(80, 20)
        self.textbox5.move(880, 280)
        button17 = QPushButton("inference", self)
        button17.move(800, 320) #座標
        button17.clicked.connect(self.on_click17)


        self.show()
        
        
    @pyqtSlot()
    def on_click1(self):
        #print("button1 click") #open 1-1window
        cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
        img = cv2.imread("Q1_image/Uncle_Roger.jpg")
        width, height = img.shape[0:2]
        print (width,height)
        cv2.imshow('original image', img)

    def on_click2(self):
        #print("button2 click")
        #red
        cv2.namedWindow('red_img', cv2.WINDOW_NORMAL)
        imgr = cv2.imread("Q1_image/Flower.jpg")
        imgr[:,:,0] = 0
        imgr[:,:,1] = 0
        cv2.imshow('red_img', imgr)
        #blue
        cv2.namedWindow('blue_img', cv2.WINDOW_NORMAL)
        imgb = cv2.imread("Q1_image/Flower.jpg")
        imgb[:,:,1] = 0
        imgb[:,:,2] = 0
        cv2.imshow('blue_img', imgb)
        #green
        cv2.namedWindow('green_img', cv2.WINDOW_NORMAL)
        imgg = cv2.imread("Q1_image/Flower.jpg")
        imgg[:,:,0] = 0
        imgg[:,:,2] = 0
        cv2.imshow('green_img', imgg)

    def on_click3(self):
        #print("button3 click")
        cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
        img = cv2.imread("Q1_image/Uncle_Roger.jpg")
        cv2.imshow('original image', img)

        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        img2 = cv2.flip(img, 1)
        cv2.imshow('result', img2)

    alpha = 0.5
    beta = (1.0 - alpha)
    def on_click4(self):
        #print("button4 click") 
        img = cv2.imread("Q1_image/Uncle_Roger.jpg")
        img2 = cv2.flip(img, 1)
        cv2.namedWindow('blending result', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('blending','blending result',0,255,Blend)
        cv2.setTrackbarPos('blending','blending result',128)
        blendimg = cv2.addWeighted(img, alpha, img2, beta, 0.0)
        cv2.imshow('blending result', blendimg)
        

    def on_click5(self):
        #print("button5 click") 
        img = cv2.imread('Q2_Image/Cat.png')
        cv2.namedWindow('Median filter', cv2.WINDOW_NORMAL)
        # apply the 7x7 median filter on the image
        median = cv2.medianBlur(img,7)
        cv2.imshow('Median filter', median)

    def on_click6(self):
        #print("button6 click") 
        img = cv2.imread('Q2_Image/Cat.png')
        cv2.namedWindow('Gaussian Blur', cv2.WINDOW_NORMAL)
        # apply the 3x3 gaussian filter on the image
        gaussian = cv2.GaussianBlur(img,(3,3),0)
        cv2.imshow('Gaussian Blur', gaussian)

    def on_click7(self):
        #print("button7 click") 
        img = cv2.imread('Q2_Image/Cat.png')
        cv2.namedWindow('Bilateral filter', cv2.WINDOW_NORMAL)
        # apply the 3x3 gaussian filter on the image
        bilateral = cv2.bilateralFilter(img,9,90,90)
        cv2.imshow('Bilateral filter', bilateral)

    def on_click8(self):
        #print("button8 click") 
        img = cv2.imread('Q3_Image/Chihiro.jpg', cv2.IMREAD_GRAYSCALE)
        cv2.namedWindow('gaussian', cv2.WINDOW_NORMAL)
        out=Gaussian(img)
        cv2.imshow('gaussian',out)

    def on_click9(self):
        #print("button9 click") 
        img = cv2.imread('Q3_Image/Chihiro.jpg', cv2.IMREAD_GRAYSCALE)
        out=Gaussian(img)
        cv2.namedWindow('Sobelx', cv2.WINDOW_NORMAL)
        out_sobelx = Sobelx(out)
        cv2.imshow('Sobelx', out_sobelx)

    def on_click10(self):
        #print("button10 click") 
        img = cv2.imread('Q3_Image/Chihiro.jpg', cv2.IMREAD_GRAYSCALE)
        out=Gaussian(img)
        cv2.namedWindow('Sobely', cv2.WINDOW_NORMAL)
        out_sobely = Sobely(out)
        cv2.imshow('Sobely', out_sobely)

    def on_click11(self):
        #print("button11 click") 
        img = cv2.imread('Q3_Image/Chihiro.jpg', cv2.IMREAD_GRAYSCALE)
        out=Gaussian(img)
        cv2.namedWindow('Magnitude', cv2.WINDOW_NORMAL)
        out_sobel = Sobel(out)
        cv2.imshow('Magnitude', out_sobel)
     
    def on_click12(self):
        #print("button12 click") 
        angle = float(self.textbox1.text())
        #print (angle)
        scale = float(self.textbox2.text())
        #print (scale)
        tx = float(self.textbox3.text())
        #print (tx)
        ty = float(self.textbox4.text())
        #print (ty)
        cv2.namedWindow('4', cv2.WINDOW_NORMAL)
        img = cv2.imread('Q4_Image/Parrot.png')
        cv2.imshow('4', img)

        cv2.namedWindow('4ans', cv2.WINDOW_NORMAL)
        rows, cols = img.shape[0:2]
        M1 = np.float32([[1,0,tx],[0,1,ty]])
        dst = cv2.warpAffine(img, M1, (cols,rows)) #move
        M2 = cv2.getRotationMatrix2D((cols/2,rows/2),angle, scale) #rotate and scale
        dst = cv2.warpAffine(dst, M2, (cols, rows))
        cv2.imshow('4ans', dst)

    def on_click13(self):
        print("button13 click")
    def on_click14(self):
        print("button14 click")
    def on_click15(self):
        print("button15 click")
    def on_click16(self):
        print("button16 click")
    def on_click17(self):
        print("button17 click")
    
class Smallwindow(QWidget):
    def __init__(self,name):
        super().__init__()
        self.title = name
        self.left = 100
        self.top = 100
        self.width = 300
        self.height = 300
        self.initUI()
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

def Newshow():
    global alpha,beta
    img = cv2.imread("Q1_image/Uncle_Roger.jpg")
    img2 = cv2.flip(img, 1)
    blendimg = cv2.addWeighted(img, alpha, img2, beta, 0.0)
    cv2.imshow('blending result', blendimg)
    #print(alpha,beta)
    cv2.setTrackbarPos('blending','blending result',int(alpha*255))

def Blend(x):
    global alpha,beta
    valtotal = cv2.getTrackbarPos('blending','blending result')
    #print(valtotal)
    alpha=valtotal/255
    beta = (1.0 - alpha)
    Newshow()

def Gaussian(img):
    #print(img.shape)
    kernel=np.array([[0.045,0.122,0.045],[0.122,0.332,0.122],[0.045,0.122,0.045]])
    result = cv2.filter2D(img,-1,kernel)
    #result = result*255.0 / result.max() #正規化到0~255    
    return np.uint8(result) #像素值0~255 -> unit8

def Sobelx(img):
    kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    result = cv2.filter2D(img,-1,kernel)
    #result = result*255.0 / result.max() #正規化到0~255    
    return np.uint8(result) #像素值0~255 -> unit8

def Sobely(img):
    kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) 
    result = cv2.filter2D(img,-1,kernel)
    #result = result*255.0 / result.max() #正規化到0~255
    return np.uint8(result) #像素值0~255 -> unit8

def Sobel(img):
    sobel_X = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])  
    sobel_Y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) 
    #計算在橫&縱的梯度 也就是和sobel_X、sobel_Y的捲積
    result1 = cv2.filter2D(img,-1,sobel_X)
    result2 = cv2.filter2D(img,-1,sobel_Y)
    result1 = result1*255.0 / result1.max()
    result2 = result2*255.0 / result2.max()
    magnitude = (result1**2+result2**2)**0.5 
    magnitude = magnitude*255.0 / magnitude.max() #正規化到0~255
    return np.uint8(magnitude) #像素值0~255 -> unit8

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window =Mainwindow()
    window.show()
    sys.exit(app.exec_())
    
