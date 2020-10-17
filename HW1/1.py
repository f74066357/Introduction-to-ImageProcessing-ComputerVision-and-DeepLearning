# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 10:47:42 2020

@author: yes19
"""
import sys
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class Mainwindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "F7406637_HW1"
        self.left = 50
        self.top = 50
        self.width = 720
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
        textbox1 = QLineEdit(self)
        textbox1.resize(80, 20)
        textbox1.move(560, 80)
        label6 = QLabel(self)
        label6.setText("deg") 
        label6.move(650, 80)
        label7 = QLabel(self)
        label7.setText("Scaling:") 
        label7.move(500, 130)
        textbox2 = QLineEdit(self)
        textbox2.resize(80, 20)
        textbox2.move(560, 130)
        label8 = QLabel(self)
        label8.setText("Tx:") 
        label8.move(500, 180)
        textbox3 = QLineEdit(self)
        textbox3.resize(80, 20)
        textbox3.move(560, 180)
        label9 = QLabel(self)
        label9.setText("pixel") 
        label9.move(650, 180)
        label10 = QLabel(self)
        label10.setText("Ty:") 
        label10.move(500, 230)
        textbox4 = QLineEdit(self)
        textbox4.resize(80, 20)
        textbox4.move(560, 230)
        label11 = QLabel(self)
        label11.setText("pixel") 
        label11.move(650, 230)

        button12 = QPushButton("4 Transformation", self)
        button12.move(500, 280) #座標
        button12.clicked.connect(self.on_click12)

        self.show()
        
        
    """创建鼠标点击事件"""
    @pyqtSlot()
    def on_click1(self):
        print("button1 click") #open 1-1window
        cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
        img = cv2.imread("Q1_image/Uncle_Roger.jpg")
        width, height = img.shape[0:2]
        print (width,height)
        cv2.imshow('original image', img)

    def on_click2(self):
        print("button2 click")
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
        print("button3 click")
        cv2.namedWindow('original image', cv2.WINDOW_NORMAL)
        img = cv2.imread("Q1_image/Uncle_Roger.jpg")
        cv2.imshow('original image', img)

        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        img2 = cv2.flip(img, 1)
        cv2.imshow('result', img2)

    def on_click4(self):
        print("button4 click") 

    def on_click5(self):
        print("button5 click") 

    def on_click6(self):
        print("button6 click") 

    def on_click7(self):
        print("button7 click") 

    def on_click8(self):
        print("button8 click") 

    def on_click9(self):
        print("button9 click") 

    def on_click10(self):
        print("button10 click") 

    def on_click11(self):
        print("button11 click") 
     
    def on_click12(self):
        print("button12 click") 
    
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window =Mainwindow()
    window.show()
    sys.exit(app.exec_())
    
