import tensorflow as tf
import numpy as np
import sys
import copy
import src.facenet
import cv2
import os
import src.align.detect_face
import camera_setting
import matplotlib.pyplot as plt

if __name__ == '__main__':

    modelpath = "E:\\leju_src\\Face-collect\\models"

    MAX_DISTINCT=1.22
    minsize = 15  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    image_size=120
    margin=32
    gpu_memory_fraction=1.0


    #创建网络
    with tf.Graph().as_default():
        gpu_memory_fraction = 1.0
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = src.align.detect_face.create_mtcnn(sess, None)


    with tf.Graph().as_default():
        # 加载模型
        sess = tf.Session()
        # src.facenet.load_model(modelpath)
        # 加载模型
        meta_file, ckpt_file = src.facenet.get_model_filenames(modelpath)
        saver = tf.train.import_meta_graph(os.path.join(modelpath, meta_file))
        saver.restore(sess, os.path.join(modelpath, ckpt_file))

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # 进行人脸识别，加载
        print('loading parameters')

        capture_interval = 5
        capture_count = 0
        frame_count = 0
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
        imgpath="E:/leju_src/pythonopencv/img/{0}.jpg"

        windowsname = "face"
        cv2.namedWindow(windowsname,0)
        cv2.resizeWindow(windowsname, 1920, 300)
        cv2.moveWindow(windowsname, 0, 1080-380) 
        windowsshow = np.zeros([300, 1920,3], np.uint8)
        windows_num = 0
        windows_x = 0
        windows_y = 0

        while True: 
            #opencv读取图片，开始进行人脸识别
            ret, frame = cap.read()
            image = camera_setting.undistorted(frame)
            cv2.imshow('face_img',image)

            # 设置默认插入时 detect_multiple_faces =Flase只检测图中的一张人脸，True则检测人脸中的多张
            #一般入库时只检测一张人脸，查询时检测多张人脸
            detect_multiple_faces=True
            img = image.copy()
            bounding_boxes, _ = src.align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            print('找到人脸数目为：{}'.format(nrof_faces))

            if nrof_faces > 0:
                capture_count += 1
                det = bounding_boxes[:, 0:4]
                det_arr = []
                img_size = np.asarray(img.shape)[0:2]
                if nrof_faces > 1:
                    if detect_multiple_faces:
                        for i in range(nrof_faces):
                            det_arr.append(np.squeeze(det[i]))
                    else:
                        bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                        img_center = img_size / 2
                        offsets = np.vstack(
                            [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                        index = np.argmax(
                            bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                        det_arr.append(det[index, :])
                else:
                    det_arr.append(np.squeeze(det))

                images = np.zeros((len(det_arr), image_size, image_size, 3))
                recimg = img.copy()
                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                    cv2.imshow("face{0}".format(i),cropped)
                    cv2.rectangle(recimg,(bb[0],bb[1]),(bb[2],bb[3]),(255,0,0),2) 
                    scaled = cv2.resize(cropped, (image_size, image_size))
                    images[i] = scaled
                cv2.imshow("rectangle",recimg)
            
            if nrof_faces > 0:
                images = images
            else:
                # 如果没有检测到人脸  直接返回一个1*3的0矩阵  多少维度都行  只要能和是不是一个图片辨别出来就行
                images = np.zeros((1, 3))
            if len(images.shape) < 4:
                capture_count = 0
                print("NO FACE")
            else:
                #5帧都采集到人脸
                if(capture_count%capture_interval == 0):
                    for i in range (len(images)):
                        rimg = images[i].copy()
                        cv2.imshow("face{0}".format(i),rimg)
                        imgInfo = rimg.shape
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
                                windowsshow[(windows_x+i),(windows_y+j)]=rimg[i,j]
                        cv2.imshow(windowsname,windowsshow)
                        windows_num += 1
            input = cv2.waitKey(1) & 0xFF
            if input == ord('q'): 
                cap.release()
                break
        cv2.destroyAllWindows()






