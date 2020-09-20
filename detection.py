#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ScriptName  : detection.py
@Project     : Object-Detection-YOLOv3
@Author      : Meng Peng
@Date        : 22-08-2020
@Description : detect and track objects using YOLOv3 with tensorflow and tensornets
"""
import cv2
import time
import numpy as np
import tensornets as nets
import tensorflow as tf


class ObjectDetector:
    __inputs = None
    __model = None
    __sess = None
    __classes = None
    __list_of_classes = None

    def __init__(self):
        self.yolo()

    def yolo(self):
        self.__inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
        self.__model = nets.YOLOv3COCO(self.__inputs, nets.Darknet19)
        self.__classes = {'0': 'person', '1': 'bicycle', '2': 'car',
                          '3': 'bike', '5': 'bus', '7': 'truck'}
        self.__list_of_classes = [0, 1, 2, 3, 5, 7]
        # to display other detected objects,
        # change the classes and list of classes to their respective COCO
        # indices available in the website.
        # Here 0th index is for people and 1 for bicycle and so on.
        # If you want to detect all the classes, add the indices to this list
        try:
            self.__sess = tf.Session()
            self.__sess.run(self.__model.pretrained())
        except Exception as e:
            print(e)

    def detection(self, img):
        img_height, img_width = img.shape[:2]

        img_norm = cv2.resize(img, (416, 416))
        img_preds = np.array(img_norm).reshape(-1, 416, 416, 3)

        start_time = time.time()

        preds = self.__sess.run(self.__model.preds,
                                {self.__inputs: self.__model.preprocess(img_preds)})

        print("--- %s seconds ---" % (time.time() - start_time))  # to time it

        boxes = self.__model.get_boxes(preds, img_preds.shape[1:3])
        boxes1 = np.array(boxes, dtype=object)

        for j in self.__list_of_classes:  # iterate over classes
            count = 0
            label = ''
            if str(j) in self.__classes:
                label = self.__classes[str(j)]
            if len(boxes1) != 0:
                # iterate over detected vehicles
                for i in range(len(boxes1[j])):
                    box = boxes1[j][i]
                    # setting confidence threshold as 50%
                    if box[4] >= 0.50:
                        count += 1
                        cv2.rectangle(img,
                                      (int(box[0] * img_width/416), int(box[1] * img_height/416)),
                                      (int(box[2] * img_width/416), int(box[3] * img_height/416)),
                                      (0, 255, 0), 2)
                        cv2.putText(img, label,
                                    (int(box[0] * img_width / 416), int(box[1]*img_height / 416)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), lineType=cv2.LINE_AA)
            print(label, ": ", count)
        return img

    def stop(self):
        if self.__sess:
            self.__sess.close()
