from insectvision.net.willshawnet import WillshawNet
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

nn = WillshawNet(nb_channels=1)
nn.update = True
for i in range(5):
    img = cv.imread('Vision/img/image_'+ str(i) + '.jpg',cv.IMREAD_GRAYSCALE)
    img = cv.resize(img,(60,6), interpolation=cv.INTER_AREA)
    en = nn(img.flatten())
nn.update = False

cap = cv.VideoCapture(0)
en_output = []
frame_list = []
image_num = 0
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
    frame_list.append(gray)

    # image preprocess
    img = cv.resize(gray,(60,6), interpolation=cv.INTER_AREA)
    en = nn(img.flatten())
    en_output.append(en[0])

    k = cv.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('c'):
        print('saving image')
        cv.imwrite('Vision/img/image_{}.jpg'.format(image_num),gray)
        image_num += 1


x = np.arange(len(en_output))
plt.plot(x, en_output)
plt.xlabel("frame")
plt.ylabel("familiarity")
plt.show()

i = np.argmax(en_output)
cv.imshow('111',frame_list[i])
cv.waitKey(0)