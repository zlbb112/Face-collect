#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import copy
import camera_setting
import numpy as np
import sys
import scipy as sp   ##在numpy基础上实现的部分算法库
import matplotlib.pyplot as plt  ##绘图库
import os
from PIL import Image
import threading
from time import ctime,sleep

def get_image(path):
    #获取图片
    img=cv2.imread(path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    return img, gray
    
def Gaussian_Blur(gray):
    # 高斯去噪
    blurred = cv2.GaussianBlur(gray, (9, 9),0)
    
    return blurred
    
def Sobel_gradient(blurred):
    # 索比尔算子来计算x、y方向梯度
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
    
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    
    return gradX, gradY, gradient

def Thresh_and_blur(gradient):
    
    blurred = cv2.GaussianBlur(gradient, (9, 9),0)
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    
    return thresh
    
def image_morphology(thresh):
    # 建立一个椭圆核函数
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25),(-1,-1))#矩形：MORPH_RECT 交叉形：MORPH_CORSS 椭圆形：MORPH_ELLIPSE
    # 执行图像形态学, 细节直接查文档，很简单
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    
    return closed
    
def findcnts_and_box_point(closed):
    # 这里opencv3返回的是三个参数
    (_, cnts, _) = cv2.findContours(closed.copy(), 
        cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # compute the rotated bounding box of the largest contour
    #参数一： 二值化图像
    #参数二：轮廓类型
    # cv2.RETR_EXTERNAL,             #表示只检测外轮廓
    # cv2.RETR_CCOMP,                #建立两个等级的轮廓,上一层是边界
    # cv2.RETR_LIST,                 #检测的轮廓不建立等级关系
    # cv2.RETR_TREE,                 #建立一个等级树结构的轮廓
    # cv2.CHAIN_APPROX_NONE,         #存储所有的轮廓点，相邻的两个点的像素位置差不超过1
    #参数三：处理近似方法
    # cv2.CHAIN_APPROX_SIMPLE,         #例如一个矩形轮廓只需4个点来保存轮廓信息
    # cv2.CHAIN_APPROX_TC89_L1,
    # cv2.CHAIN_APPROX_TC89_KCOS
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    
    return box

def drawcnts_and_cut(original_img, box):
    # 因为这个函数有极强的破坏性，所有需要在img.copy()上画
    # draw a bounding box arounded the detected barcode and display the image
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)
    
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    crop_img = original_img[y1:y1+hight, x1:x1+width]
    
    return draw_img, crop_img

def face_img(image,gray,face_cascade): 
    # 探测图片中的人脸
    faces = face_cascade.detectMultiScale(gray ,1.2,3)  # (32, 32)
    print ("Found {0} faces!".format(len(faces)))
    # windows_x = 0
    # windows_y = 0
    # windows_num = 0
    # for windows_num in range(len(faces)):
    #     windows_x = windows_num % 3 
    #     windows_y = int(windows_num / 3)
    #     windowsname = "face num {0}".format(windows_num)
    #     cv2.namedWindow(windowsname,0)
    #     cv2.resizeWindow(windowsname, 640, 360)
    #     cv2.moveWindow(windowsname, 640*windows_x, 360*windows_y)
    windowsname = "face"
    cv2.namedWindow(windowsname,0)
    cv2.resizeWindow(windowsname, 1920, 300)
    cv2.moveWindow(windowsname, 0, 1080-400)  #380
    windows_num = 0
    windows_len = 0
    windowsshow = 10 * np.ones([300, 1920,3], np.uint8)
    for (x,y,w,h) in faces:
        # windowsname = "face num {0}".format(windows_num)
        windowsface = image[y-30:y+h+10, x-10:x+w+10]
        smallface = cv2.resize(windowsface,(120,120))
        # cv2.imshow("faceonly{0}".format(windows_num),smallface)
        #  
        ##############图像融合#############
        # smallface_mask = 255 * np.ones(smallface.shape, smallface.dtype)  #效果不好
        # poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
        # cv2.fillPoly(smallface_mask, [poly], (255, 255, 255))
        # windows_x = int((120*(windows_num%16)+10+120)/2)
        # windows_y = int((20+120*(windows_num/16)+(windows_num/16)*20+120)/2)
        # imgout = cv2.seamlessClone(smallface,windowsshow,smallface_mask,(windows_x,windows_y),cv2.MIXED_CLONE)
        imgInfo = smallface.shape
        height= imgInfo[0]
        width = imgInfo[1]
        deep = imgInfo[2]
        windows_x = 20+140*int(windows_num/14)
        windows_y = 20+120*(windows_num%14)+int(windows_num%14)*10
        if windows_num >= 28:
            windows_num = 0
            windows_x = 20+120*int(windows_num/14)
            windows_y = 20+120*(windows_num%14)+int(windows_num%14)*10
            # windows_len += 1
            # windows_x = 20+120*int(windows_num/15)-280*windos_len
            # windows_y = 120*(windows_num%14)+int((windows_num+1)%15)*10 -1820*windos_len
        elif (windows_y/1790) >1:
            windows_y = windows_y - 1840
            windows_len =1

        for i in range(height):
            for j in range(width):
                windowsshow[(windows_x+i),(windows_y+j)]=smallface[i,j]
                # # windowsshow[i,j]=smallface[i,j]
                # if j>118:
                #     cv2.imshow(windowsname,windowsshow)
                #     cv2.waitKey(0)
        cv2.imshow(windowsname,windowsshow)
        windows_num += 1
        cv2.waitKey(0)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2) 
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = image[y:y+h, x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image


def face_camera(image,gray,face_cascade): 
    # 探测视频一帧中的人脸
    faces = face_cascade.detectMultiScale(gray ,1.2,3)  # (32, 32)
    print ("Found {0} faces!".format(len(faces)))
    windowsname = "face"
    cv2.namedWindow(windowsname,0)
    cv2.resizeWindow(windowsname, 1920, 300)
    cv2.moveWindow(windowsname, 0, 1080-380) 
    
    windowsshow = np.zeros([300, 1920,3], np.uint8)
    for (x,y,w,h) in faces:
        windowsface = image[y-30:y+h+10, x-10:x+w+10]
        smallface = cv2.resize(windowsface,(120,120))
        imgInfo = smallface.shape
        height= imgInfo[0]
        width = imgInfo[1]
        deep = imgInfo[2]
        windows_x = 20+140*int(windows_num/14)
        windows_y = 20+120*(windows_num%14)+int(windows_num%14)*10
        if windows_num >= 28:
            windows_num = 0
            windows_x = 20+120*int(windows_num/14)
            windows_y = 20+120*(windows_num%14)+int(windows_num%14)*10

        elif (windows_y/1790) >1:
            windows_y = windows_y - 1840

        for i in range(height):
            for j in range(width):
                windowsshow[(windows_x+i),(windows_y+j)]=smallface[i,j]
        cv2.imshow(windowsname,windowsshow)
        windows_num += 1

    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2) 
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = image[y:y+h, x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return image

def video_demo(cameranum,adjust,face_cascade):
    # 抓取摄像头视频图像,0是代表摄像头编号，只有一个的话默认为0
    cap = cv2.VideoCapture(cameranum)  #创建内置摄像头变量
    sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(sz)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow("camera_img_distor",0)
    cv2.resizeWindow("camera_img_distor", 640, 480)
    cv2.moveWindow('camera_img_distor',200,100)
    cv2.namedWindow("face_img",0)
    cv2.resizeWindow("face_img", 640, 480)
    cv2.moveWindow('face_img',840,100)
    windows_num =0
    while(True):
        ref,frame=cap.read()
        if ref == True:
            if adjust == 1:
                #矫正摄像头参数
                camera_img_distor = camera_setting.undistorted(frame)
            elif adjust == 0:
                camera_img_distor = frame
            gray  = cv2.cvtColor(camera_img_distor,cv2.COLOR_BGR2GRAY)
            cv2.imshow('camera_img_distor',camera_img_distor)
            image=face_camera(camera_img_distor,gray,face_cascade)
            cv2.imshow('face_img',image)
        elif ref == False:
            print("camera erro!")

        input = cv2.waitKey(1) & 0xFF
        if input == ord('s'):  #更换存储目录
            # class_name = input("请输入存储目录：")
            # while os.path.exists(class_name):
            #     class_name = input("目录已存在！请输入存储目录：")
            # os.mkdir(class_name)
            cap.release()
            break
        elif input == ord("q"):
            cap.release()
            break
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':

    # imgpath="E:/leju_src/pythonopencv/img/"
    # immnum = 0
    # imglen = 5

    # 获取训练好的人脸的参数数据，这里直接从GitHub上使用默认值
    face_alt = cv2.CascadeClassifier("E:/leju_src/pythonopencv/xml/haarcascade_frontalface_alt.xml") #效果最好
    face_alt2 = cv2.CascadeClassifier("E:/leju_src/pythonopencv/xml/haarcascade_frontalface_alt2.xml")
    face_default = cv2.CascadeClassifier("E:/leju_src/pythonopencv/xml/haarcascade_frontalface_default.xml")
    face_cascade = face_alt   
    # eye_default = cv2.CascadeClassifier("E:/leju_src/pythonopencv/xml/haarcascade_eye.xml ")
    # eye_cascade = eye_default

    # video_demo(0,1,face_cascade)

    # 抓取摄像头视频图像,0是代表摄像头编号，只有一个的话默认为0
    cap = cv2.VideoCapture(0)  #创建内置摄像头变量
    sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(sz)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow("face_img",0)
    cv2.resizeWindow("face_img", 640, 480)
    cv2.moveWindow('face_img',420,100)
    
    windowsname = "face"
    cv2.namedWindow(windowsname,0)
    cv2.resizeWindow(windowsname, 1920, 300)
    cv2.moveWindow(windowsname, 0, 1080-380) 
    windows_num =0
    adjust = 1
    windows_len =0
    windowsshow = np.zeros([300, 1920,3], np.uint8)

    while(True):
        ref,frame=cap.read()
        if ref == True:
            if adjust == 1:
                #矫正摄像头参数
                image = camera_setting.undistorted(frame)
            elif adjust == 0:
                image = frame
            gray  = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray ,1.2,3)  # (32, 32)
            print ("Found {0} faces!".format(len(faces)))
            windows_len += 1
            if windows_len >10:
                windows_len = 0
                for (x,y,w,h) in faces:
                    windowsface = image[y-30:y+h+10, x-10:x+w+10]
                    smallface = cv2.resize(windowsface,(120,120))
                    imgInfo = smallface.shape
                    height= imgInfo[0]
                    width = imgInfo[1]
                    windows_x = 20+140*int(windows_num/14)
                    windows_y = 20+120*(windows_num%14)+int(windows_num%14)*10
                    if windows_num >= 28:
                        windows_num = 0
                        windows_x = 20+120*int(windows_num/14)
                        windows_y = 20+120*(windows_num%14)+int(windows_num%14)*10

                    elif (windows_y/1790) >1:
                        windows_y = windows_y - 1840

                    for i in range(height):
                        for j in range(width):
                            windowsshow[(windows_x+i),(windows_y+j)]=smallface[i,j]
                    cv2.imshow(windowsname,windowsshow)
                    windows_num += 1

            for (x,y,w,h) in faces:
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2) 
        elif ref == False:
            print("camera erro!")
        cv2.imshow('face_img',image)
        input = cv2.waitKey(1) & 0xFF
        if input == ord('s'):  #更换存储目录
            # class_name = input("请输入存储目录：")
            # while os.path.exists(class_name):
            #     class_name = input("目录已存在！请输入存储目录：")
            # os.mkdir(class_name)
            cap.release()
            break
        elif input == ord("q"):
            cap.release()
            break
    cv2.destroyAllWindows()




# video_demo(0,imgPath)

