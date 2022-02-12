import numpy as np
import cv2 as cv

try:
    img = cv.imread('img_1.jpg',0)
    crop_img = img[100:370, :]
    cv.imshow("cropped", crop_img)
    cv.waitKey(0)
except KeyboardInterrupt:
    print("end")
