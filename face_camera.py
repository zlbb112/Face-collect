#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import copy
import camera_setting
import numpy as np
import sys

if __name__ == '__main__':

    # 获取训练好的人脸的参数数据，这里直接从GitHub上使用默认值
    face_alt = cv2.CascadeClassifier("E:/leju_src/pythonopencv/xml/haarcascade_frontalface_alt.xml") #效果最好
    face_alt2 = cv2.CascadeClassifier("E:/leju_src/pythonopencv/xml/haarcascade_frontalface_alt2.xml")
    face_default = cv2.CascadeClassifier("E:/leju_src/pythonopencv/xml/haarcascade_frontalface_default.xml")
    face_cascade = face_alt   
    eye_default = cv2.CascadeClassifier("E:/leju_src/pythonopencv/xml/haarcascade_eye.xml ")
    eye_cascade = eye_default

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
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = image[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

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
        if input == ord('q'): 
            cap.release()
            break
    cv2.destroyAllWindows()


