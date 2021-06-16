from insectvision.net.willshawnet import WillshawNet
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class Vision:
    def __init__(self):
        self.nn = WillshawNet(nb_channels=1)

        self.num_img_captured = 0

    def preprocess(sefl,image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        p_img = cv.resize(gray,(60,6), interpolation=cv.INTER_AREA)
        return p_img
    
    def start_learning(self):
        self.nn.update = True
        for i in range(self.num_img_captured):
            img = cv.imread('Vision/img/image_'+ str(i) + '.jpg',cv.IMREAD_GRAYSCALE)
            en = self.nn(img.flatten())
        self.nn.update = False
    
    def capture_image(self):
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        ret, frame = cap.read()
        cv.imwrite('Vision/img/image_{}.jpg'.format(self.num_img_captured),frame)
        self.num_img_captured += 1
        cap.release()
    
    def start_streaming(self):
        cap = cv.VideoCapture(0)
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # image preprocess
            
            # Display the resulting frame
            cv.imshow('frame', frame)

            k = cv.waitKey(1)
            if k == ord('q'):
                break
            elif k == ord('c'):
                cv.imwrite('Vision/img/image_{}.jpg'.format(self.num_img_captured),frame)
                self.num_img_captured += 1
                print('frame saved')
    

    