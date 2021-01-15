# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:55:25 2020

@author: yes19
"""
import sys
import cv2 as cv
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *\

import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import random

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense,Dropout
from keras.models import Model

class Mainwindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "F7406637_HW1"
        self.left = 50
        self.top = 50
        self.width = 300
        self.height = 360
        self.initUI()
    
    def initUI(self):
        
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        label12 = QLabel(self)
        label12.setText("5. VGG16") 
        label12.move(50, 30)
        button13 = QPushButton("1. show train images", self)
        button13.move(50, 80) #座標
        button13.clicked.connect(self.on_click13)
        button14 = QPushButton("2. show hyperparameters", self)
        button14.move(50, 130) #座標
        button14.clicked.connect(self.on_click14)
        button15 = QPushButton("3. show model structure", self)
        button15.move(50, 180) #座標
        button15.clicked.connect(self.on_click15)
        button16 = QPushButton("4. show accuracy", self)
        button16.move(50, 230) #座標
        button16.clicked.connect(self.on_click16)
        label17 = QLabel(self)
        label17.setText("5. test image index:") 
        label17.move(50, 280)
        self.textbox5 = QLineEdit(self)
        self.textbox5.resize(80, 20)
        self.textbox5.move(180, 280)
        button17 = QPushButton("inference", self)
        button17.move(100, 320) #座標
        button17.clicked.connect(self.on_click17)

        self.show()
        
        
    @pyqtSlot()
    def on_click13(self):
        #print("button13 click")
        (x,y),(x_test,y_test) = keras.datasets.cifar10.load_data()
        classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        fig, axes = plt.subplots(1,10)
        for i in range(10):
            r=random.randint(0,len(x))
            #print(classes[int(y[r])]+' ' ,end='')
            axes[i].imshow(x[r])
            axes[i].set_xlabel(classes[int(y[r])])
        plt.show()

    def on_click14(self):
        #print("button14 click")
        print('hyperparameters:')
        print('batch size: 32')
        print('learning rate: 0.0001')
        print('optimizer: SGD')

    def on_click15(self):
        #print("button15 click")

        model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
        #model_vgg16_conv.summary()

        input = Input(shape=(32,32,3),name = 'image_input')

        #Use the generated model 
        output_vgg16_conv = model_vgg16_conv(input)

        #Add the fully-connected layers 
        x = Flatten(name='flatten')(output_vgg16_conv)
        x = Dense(256, activation='relu', name='dense_1')(x)
        x = Dense(256, activation='relu', name='dense_2')(x)
        x = Dense(10, activation='softmax', name='predictions')(x)
        
        my_model=Model(input,x)
        #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
        my_model.summary()

    def on_click16(self):
        #print("button16 click")
        img1 = cv.imread('accuracy.png', cv.IMREAD_UNCHANGED)
        img2 = cv.imread('loss.png', cv.IMREAD_UNCHANGED)
        cv.namedWindow('Result')
        imgs = np.hstack([img1,img2])
        cv.imshow('Result', imgs)
        plt.show()

    def on_click17(self):
        #print("button17 click")

        k=int(self.textbox5.text())
        print(k)
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        plt.figure()
        plt.imshow(x_test[k])
        plt.show()

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
        input = Input(shape=(32,32,3),name = 'image_input')

        #Use the generated model 
        output_vgg16_conv = model_vgg16_conv(input)

        #Add the fully-connected layers 
        x = Flatten(name='flatten')(output_vgg16_conv)
        x = Dense(256, activation='relu', name='dense_1')(x)
        x = Dense(256, activation='relu', name='dense_2')(x)
        x = Dense(10, activation='softmax', name='predictions')(x)

        #Create your own model 
        my_model=Model(input,x)
        my_model.load_weights('train.h5')
        image = np.expand_dims(x_test[k], axis=0)
        predict_t = my_model.predict(image)
        #print(predict_t)
        bar = predict_t[0]

        plt.figure()
        tick_label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        plt.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], height=bar, width=0.5)
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], tick_label)
        plt.show()
    
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
    
