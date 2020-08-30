#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ScriptName  : 
@Project     : Object-Detection-YOLOv3
@Author      : Meng Peng
@Date        : 22-08-2020
@Description : input image or video, test object detection function in detection.py
"""
import os
import cv2
from detection import ObjectDetector


def get_base_path():
    base_path = os.path.abspath(".")
    return base_path


def get_file_path(filename):
    full_path = get_base_path() + filename
    return full_path


if __name__ == '__main__':
    detector = ObjectDetector()

    cv2.namedWindow('display', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('display', 960, 720)

    # test image
    image_path = get_file_path("/data/image.jpg")
    image = cv2.imread(image_path)
    img_detect = detector.detection(image)
    cv2.imwrite(get_base_path()+"/data/image_detect.jpg", img_detect)
    # Display the output
    cv2.imshow("display", img_detect)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    '''
    
    # test video
    video_path = get_file_path("/data/video1.mp4")
    cap = cv2.VideoCapture(video_path)
    # change the path to your directory or to '0' for webCam
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(get_base_path() + "/data/video1_detect.mp4",
                          fourcc, fps, size)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_detect = detector.detection(frame)
            out.write(frame_detect)
            # Display the output
            cv2.imshow("display", frame_detect)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            out.release()
            cap.release()
    cv2.destroyAllWindows()
    '''

    detector.stop()
