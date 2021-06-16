import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
image_num = 0
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    '''
    image preprocess
    '''

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', gray)
    k = cv.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('c'):
        print('saving image')
        cv.imwrite('Vision/img/image_{}.jpg'.format(image_num),gray)
        image_num += 1
        
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()