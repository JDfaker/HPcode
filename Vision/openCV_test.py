import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('img/pig_0.jpg',0)

def shift_img(img, pixel):
    left_img = img[0:img.shape[0], 0: angle2pixels]
    right_img = img[0:img.shape[0], angle2pixels+1: img.shape[1]]
    return np.concatenate((right_img,left_img),axis=1)

def resize(img, width, height):
    return cv2.resize(img,(74,19), interpolation=cv2.INTER_AREA)

#plot gray scale
plt.imshow(rotated_img,cmap=plt.cm.gray)
plt.show()

# #matplot is using BGR order instead of RGB
# plt.imshow(resized_img[:,:,::-1])
# plt.show()