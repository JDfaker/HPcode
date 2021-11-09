import sys
sys.path.append(r"C:\\Users\\JD\\Desktop\\HPcode\\ricoh-theta-module-master\\")

from thetacv.capture.stream import Frame, Camera
from thetacv.capture.preprocess import NonePreprocess, Dewarp, Panorama

import cv2 as cv
import numpy as np

preprocessor = Panorama(colour_spec=False, bw=False, rebuild_map=True)
stream = Camera(preprocessor=preprocessor)

num_img_captured = 0
img_size = (1024, 512)
while True:
    # get the new frame
    frame = stream.next()

    # place side by side the left and right eye perceived images
    img = np.concatenate((frame.leye, frame.reye), axis=1)

    # read actual image size
    original_img_size = np.asarray(img.shape[:2][::-1],\
                                                    dtype=np.float32)

    # compute the ratio between the real image and the previewed
    fx, fy = np.asarray(img_size).astype(float) / \
                                        original_img_size.astype(float)
    
    img = cv.resize(img, (0, 0), fx=fx, fy=fy)
    # show image
    cv.imshow('Live', img)

    k = cv.waitKey(1)
    if k == ord('q'):
        cv.destroyWindow('frame')
        break
    elif k == ord('c'):
        cv.imwrite('img/image_{}.jpg'.format(num_img_captured),img)
        print('frame saved')