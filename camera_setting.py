#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" camera 模块 fisheye 鱼眼相机校正"""
import numpy as np
import cv2
import copy


def undistorted(image_pic):
    img = image_pic.copy()
    _img_shape = img.shape[:2]
    DIM = _img_shape[::-1]
    # print("DIM:{0}".format(DIM))
    K =np.array([[882.3622,0,688.7520],[0,883.4296,342.5895],[0,0,1]])
    D =np.array([0.1398,-0.3862,0,0])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img