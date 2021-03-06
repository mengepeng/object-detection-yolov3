# Object Detection with YOLOv3

## Objective

YOLO ("You Only Look Once") is one of the fastest object detection algorithm that uses convolutional neural networks. 
Here I use a YOLOv3 pre-trained model of tensornets for object detection and see how it works. 
This YOLOv3 model is trained on MS-COCO dataset that can detect about 90 different classes of objects like cars, trucks, bikes, people and many more!

## Dependencies & Environment
* Ubuntu 16.04
* Python 3.6
* OpenCV 4.4.0
* Tensorflow-gpu 1.15.0
* Tensornets 0.4.6

See more requirements in [environment.yml](./environment.yml). 

## Prepare

Install **tensorflow-gpu** to use gpu for calculation.
```sh
pip install tensorflow-gpu==1.15.0
```

Install **tensornets**, a PyPi library which converts various popular networks' weights into 
tensorflow's format.
```sh
pip install tensornets
```

## Quick start

Run [main.py](./main.py) to start using pre-trained YOLOv3 weights in [detection.py](./detection.py)
to detect person, bicycle, car, bike, bus and truck.
```sh
python main.py
```

![result](./data/image_detect.jpg)

## Conclusion

* The accuracy of YOLOv3 for object detection is sooooo good!
* The speed of YOLOv3 depends on gpu performance, with Nvidia Titan it can go up to 30 fps.
