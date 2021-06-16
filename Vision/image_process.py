import numpy as py
import cv2 as cv

class Vision:
    def __init__(self):
        pass
    def capture_img(self):
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        
