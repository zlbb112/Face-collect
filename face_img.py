#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import copy
import camera_setting
import numpy as np
import sys


if __name__ == '__main__':

    imgpath="E:/leju_src/pythonopencv/img/{0}.jpg"
    immnum = 0
    imglen = 5   #照片数

    # 获取训练好的人脸的参数数据，这里直接从GitHub上使用默认值
    face_alt = cv2.CascadeClassifier("E:/leju_src/pythonopencv/xml/haarcascade_frontalface_alt.xml") #效果最好
    face_alt2 = cv2.CascadeClassifier("E:/leju_src/pythonopencv/xml/haarcascade_frontalface_alt2.xml")
    face_default = cv2.CascadeClassifier("E:/leju_src/pythonopencv/xml/haarcascade_frontalface_default.xml")
    face_cascade = face_alt   
  
    windowsname = "face"
    cv2.namedWindow(windowsname,0)
    cv2.resizeWindow(windowsname, 1920, 300)
    cv2.moveWindow(windowsname, 0, 1080-380) 
    windowsshow = np.zeros([300, 1920,3], np.uint8)
    windows_num = 0
    windows_x = 0
    windows_y = 0
    for immnum in range(imglen):
        # 探测图片中的人脸
        img = cv2.imread(imgpath.format(immnum))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray ,1.2,3)  # (32, 32)
        print ("Found {0} faces!".format(len(faces)))

        for (x,y,w,h) in faces:
            face = "face num {0}".format(windows_num)
            windowsface = img[y-30:y+h+10, x-10:x+w+10]
            smallface = cv2.resize(windowsface,(120,120))
            cv2.imshow("faceonly{0}".format(windows_num),smallface)
    
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
                windows_len =1

            for i in range(height):
                for j in range(width):
                    windowsshow[(windows_x+i),(windows_y+j)]=smallface[i,j]
            cv2.imshow(windowsname,windowsshow)
            windows_num += 1
            cv2.waitKey(0)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
        cv2.imshow("img",img)
        cv2.waitKey(0)
        cv2.destroyWindow("img")
    cv2.destroyAllWindows()



