import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtGui, QtCore
import glob
import copy 
import tensorflow as tf
import datetime, os
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image
index='1'
ix=-10
iy=-10ㄕ
drawing=True

class Mainwindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "F7406637_HW2"
        self.left = 50
        self.top = 50
        self.width = 1200
        self.height = 360
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        label1 = QLabel(self)
        label1.setText("1. Find Contour") 
        label1.move(30, 30)
        button1 = QPushButton("1.1 Draw Contour", self)
        button1.move(30, 80) #座標
        button1.clicked.connect(self.on_click1)
        button2 = QPushButton("1.2 Count Coins", self)
        button2.move(30, 130) #座標
        button2.clicked.connect(self.on_click2)
        self.label2 = QLabel(self)
        self.label2.setText("There are X coins in coin01.jpg") 
        self.label2.move(30, 180)
        self.label3 = QLabel(self)
        self.label3.setText("There are X coins in coin02.jpg") 
        self.label3.move(30, 230)

        label4 = QLabel(self)
        label4.setText("2. Camera Calibration") 
        label4.move(250, 30)
        button5 = QPushButton("2.1 Corner", self)
        button5.move(250, 80) #座標
        button5.clicked.connect(self.on_click5)
        button6 = QPushButton("2.2 Intrinsic ", self)
        button6.move(250, 130) #座標
        button6.clicked.connect(self.on_click6)
        button7 = QPushButton("2.3 Extrinsic", self)
        button7.move(250, 180) #座標
        label5 = QLabel(self)
        label5.setText("select image") 
        label5.move(250, 230)

        choices = ['1', '2', '3', '4','5','6','7','8','9','10','11','12','13','14','15']
        self.combo = QComboBox(self)
        self.combo.addItems(choices)
        self.combo.move(250, 250)
        self.combo.currentIndexChanged.connect(self.onChanged)

        button7.clicked.connect(self.on_click7)
        button8 = QPushButton("2.4 Distortion", self)
        button8.move(250, 300) #座標
        button8.clicked.connect(self.on_click8)
######
        label6 = QLabel(self)
        label6.setText("3. Augmented Reality") 
        label6.move(400, 30)
        button9 = QPushButton("3.1 show", self)
        button9.move(400, 80) #座標
        button9.clicked.connect(self.on_click9)

        label4 = QLabel(self)
        label4.setText("4. Stereo Disparity") 
        label4.move(550, 30)
        label5 = QLabel(self)
        button12 = QPushButton("4.1 Stereo Disparity", self)
        button12.move(550, 80) #座標
        button12.clicked.connect(self.on_click12)
######
        label12 = QLabel(self)
        label12.setText("5. classification") 
        label12.move(800, 30)
        button13 = QPushButton("1. source code", self)
        button13.move(800, 80) #座標
        button13.clicked.connect(self.on_click13)
        button14 = QPushButton("2. screenshot of TensorBoard", self)
        button14.move(800, 130) #座標
        button14.clicked.connect(self.on_click14)
        label17 = QLabel(self)
        label17.setText("3. test image index:") 
        label17.move(800, 180)
        self.textbox5 = QLineEdit(self)
        self.textbox5.resize(80, 20)
        self.textbox5.move(930, 180)
        button15 = QPushButton("inference", self)
        button15.move(800, 230) #座標
        button15.clicked.connect(self.on_click15)
        button16 = QPushButton("4.comparison table of accuracy", self)
        button16.move(800, 280) #座標
        button16.clicked.connect(self.on_click16)

        self.show()

    @pyqtSlot()
    def onChanged(self):
        global index
        index = str(self.combo.currentText())
        #print(index)

    def on_click1(self):
        #print("button1 click") 
        image1 = cv2.imread('Datasets/Q1_Image/coin01.jpg')
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        blurred1 = cv2.GaussianBlur(gray1, (11, 11), 0)
        canny1 = cv2.Canny(blurred1, 30, 150)
        (_, cnts1, _) = cv2.findContours(canny1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours1 = image1.copy()
        result1=cv2.drawContours(contours1, cnts1, -1, (0, 255, 0), 2)
        #self.label2.setText("There are "+str(len(cnts1))+" coins in coin01.jpg")

        image2 = cv2.imread('Datasets/Q1_Image/coin02.jpg')
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        blurred2 = cv2.GaussianBlur(gray2, (11, 11), 0)
        canny2 = cv2.Canny(blurred2, 30, 150)
        (_, cnts2, _) = cv2.findContours(canny2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours2 = image2.copy()
        result2=cv2.drawContours(contours2, cnts2, -1, (0, 255, 0), 2)
        #self.label3.setText("There are "+str(len(cnts2))+" coins in coin02.jpg")
        result = np.hstack([result1, result2])
        cv2.imshow("Result1", result)

    def on_click2(self):
        #print("button2 click")
        image1 = cv2.imread('Datasets/Q1_Image/coin01.jpg')
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        blurred1 = cv2.GaussianBlur(gray1, (11, 11), 0)
        canny1 = cv2.Canny(blurred1, 30, 150)
        (_, cnts1, _) = cv2.findContours(canny1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        self.label2.setText("There are "+str(len(cnts1))+" coins in coin01.jpg")

        image2 = cv2.imread('Datasets/Q1_Image/coin02.jpg')
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        blurred2 = cv2.GaussianBlur(gray2, (11, 11), 0)
        canny2 = cv2.Canny(blurred2, 30, 150)
        (_, cnts2, _) = cv2.findContours(canny2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        self.label3.setText("There are "+str(len(cnts2))+" coins in coin02.jpg")
        
    def on_click5(self):
        #print("button5 click")
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob('Datasets/Q2_Image/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11,8), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (11,8), corners2, ret)
                cv2.namedWindow('corner',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('corner', 600,600)
                cv2.imshow('corner', img)
                cv2.waitKey(0)

    def on_click6(self):
        #print("button6 click") 
        global ret, mtx, dist, rvecs, tvecs
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob('Datasets/Q2_Image/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11,8), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
        img_size = (img.shape[1],img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        print(mtx) #camera matrix 
        
    def on_click7(self):
        #print("button7 click") 
        global ret, mtx, dist, rvecs, tvecs
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob('Datasets/Q2_Image/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11,8), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
        img_size = (img.shape[1],img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        #print(rvecs[int(index)-1]) #rotation vectors 
        #print(tvecs[int(index)-1]) #translation vectors
        rotation_mat = np.zeros(shape=(3, 3))
        R = cv2.Rodrigues(rvecs[int(index)-1], rotation_mat)[0]
        T=tvecs[int(index)-1]
        P=np.concatenate((R, T), axis=1)
        print(P)
        
    def on_click8(self):
        #print("button8 click")
        global ret, mtx, dist, rvecs, tvecs
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob('Datasets/Q2_Image/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11,8), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
        img_size = (img.shape[1],img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        print(dist) #distortion coefficients

    def on_click9(self):
        #print("button9 click") 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob('Datasets/Q3_Image/*.bmp')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11,8), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
        img_size = (img.shape[1],img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        intrinsic = np.array(mtx)
        #print(intrinsic)
        distortion = np.array(dist)
        #print(distortion)
        rotation_mat = np.zeros(shape=(3, 3))
        R = cv2.Rodrigues(rvecs[0], rotation_mat)[0]
        ext=R
        e=[]
        e.append(copy.deepcopy(ext))
        for i in range (1,5):
            R = cv2.Rodrigues(rvecs[i], rotation_mat)[0]
            np.concatenate((ext, R))
            e.append(copy.deepcopy(ext))
        #print(e)
        extrinsicl=np.array(e)
        #print(extrinsicl)
        #extrinsicl.append=cv2.Rodrigues(r[i], rotation_mat)
        #print(extrinsicl)
        extrinsicr = np.array(tvecs)
        #print(extrinsicr)
        axis = np.float32([[1, 1, 0], [5, 1, 0], [5, 5, 0], [1, 5, 0], [3, 3, -4]])
        extrinsicl = extrinsicl.astype(np.float32)
        distortion = distortion.astype(np.float32)
        intrinsic = intrinsic.astype(np.float32)
        files = glob.glob('Datasets/Q3_Image/*.bmp', recursive=True)
        while True:
            for index, images in enumerate(files):
                image = cv2.imread(images)
                re = np.zeros(shape=(3, 1))
                re = re.astype(np.float32)
                cv2.Rodrigues(extrinsicl[index], re)
                imagepoints, jac = cv2.projectPoints(axis, re, extrinsicr[index], intrinsic, distortion)
                points = []
                for point in imagepoints:
                    points.append(tuple(point.ravel()))
                image = cv2.line(image, points[0], points[4], (0, 0, 255), 10)
                image = cv2.line(image, points[1], points[4], (0, 0, 255), 10)
                image = cv2.line(image, points[2], points[4], (0, 0, 255), 10)
                image = cv2.line(image, points[3], points[4], (0, 0, 255), 10)
                image = cv2.line(image, points[0], points[1], (0, 0, 255), 10)
                image = cv2.line(image, points[1], points[2], (0, 0, 255), 10)
                image = cv2.line(image, points[2], points[3], (0, 0, 255), 10)
                image = cv2.line(image, points[3], points[0], (0, 0, 255), 10)
                image = cv2.resize(image, (512, 512))
                cv2.imshow('AR', image)
                wkey = cv2.waitKey(500) & 0xff
                if wkey == 27:
                    break
            if wkey == 27:
                break
        cv2.destroyAllWindows()

    def on_click12(self):
        #print("button12 click") 
        def draw(event,x,y,flags,param):
            global drawing
            global ix
            global iy
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing=not drawing
                ix=x
                iy=y
                cv2.rectangle(dispRGB,(ix,iy),(ix+10,iy+10),(0, 0, 255),-1)
                depth=int((2826*178)/(123+disp[y,x]))
                cv2.rectangle(dispRGB,(2100,1500),(2820,1920),(255, 255, 255),-1)
                text1='disparity:'+str(disp[y,x])+'pixel'
                text2='depth:'+str(depth)
                cv2.putText(dispRGB, text1, (2150, 1600), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(dispRGB, text2, (2150, 1800), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 0), 2, cv2.LINE_AA)
                #print(disp[y,x],depth)
        img1 = cv2.imread('Datasets/Q4_Image/imgL.png',0)
        img2 = cv2.imread('Datasets/Q4_Image/imgR.png',0)
        stereo = cv2.StereoBM_create(numDisparities=192, blockSize=27)
        disp = stereo.compute(img1, img2)
        disp = cv2.normalize(src=disp, dst=disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        disp = np.uint8(disp)
        dispRGB = cv2.cvtColor(disp,cv2.COLOR_GRAY2RGB)
        cv2.namedWindow('image',0)
        cv2.resizeWindow('image', 500, 500)
        cv2.setMouseCallback('image',draw)
        #print(dispRGB.shape)

        while(1):
            if(drawing==True):
                dispRGB = cv2.cvtColor(disp,cv2.COLOR_GRAY2RGB)
            cv2.imshow('image',dispRGB)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()

    def on_click13(self):
        print("see 'train.py'")

    def on_click14(self):
        img = cv2.imread('tensorboard.png')
        cv2.namedWindow('screenshot', cv2.WINDOW_NORMAL)
        cv2.imshow('screenshot', img)

    def on_click15(self):
        IMAGE_SIZE =(128,128)
        model_resnet = ResNet50(weights='imagenet', include_top=False)
        input = Input(shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3),name = 'image_input')
        output_resnet = model_resnet(input)
        x = Flatten(name='flatten')(output_resnet)
        x = Dense(64, activation='relu', name='dense_1')(x)
        x = Dense(64, activation='relu', name='dense_2')(x)
        x = Dense(2, activation='softmax', name='predictions')(x)
        my_model=Model(input,x)
        my_model.summary()
        my_model.load_weights('train.h5')
        cls_list = ['cats', 'dogs']
        f= self.textbox5.text()+'.jpg'
        img = image.load_img(f, target_size=(128,128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        pred = my_model.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]

        for i in top_inds:
            print('{:.3f} {}'.format(pred[i], cls_list[i]))
        for i in top_inds:
            title= cls_list[i]
            break
        img = mpimg.imread(f)
        plt.imshow(img)
        plt.axis('off') 
        plt.title('class:'+title)
        plt.show()
        
    def on_click16(self):
        plt.figure()
        tick_label = ['original', 'resize']
        plt.bar(0,height=0.8906, width=0.5)
        plt.bar(1,height=0.9297, width=0.5)
        plt.xticks([0, 1], tick_label)
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