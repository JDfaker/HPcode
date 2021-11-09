from insectvision.net.willshawnet import WillshawNet
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img= cv.imread('img/test_img/test_img.jpg')
###pre-process
img = cv.resize(img,(1000,500),interpolation=cv.INTER_AREA)
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
###

###
shifted_pixels_steps = list(range(-500,501,50))
i = 0
for step in shifted_pixels_steps:
    i+=1
    if(step >= 0):
        left_img = gray_img[0:img.shape[0], 0: step]
        right_img = gray_img[0:img.shape[0], step: 1000]
        shifted_img = np.concatenate((right_img,left_img),axis=1)
    else:
        left_img = gray_img[0:img.shape[0], 0: 1000 - abs(step)]
        right_img = gray_img[0:img.shape[0], 1000 - abs(step): 1000]
        shifted_img = np.concatenate((right_img,left_img),axis=1)
    cv.imwrite('img/test_img/{}_{}_img.png'.format(i,step),shifted_img)
###
# while(True):
#     cv.imshow('test img',gray_img)
#     k = cv.waitKey(1)
#     if k == ord('q'):
#         cv.destroyWindow('test img')
#         break