from insectvision.net.willshawnet import WillshawNet
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

num_of_pn = 2000
num_of_kc = 10000
resize_shape = (200,10)
print('building network')
nn = WillshawNet(nb_channels=1,num_pn=num_of_pn,num_kc=num_of_kc)
print('network finished')

def preprocess(image):
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    p_img = cv.resize(gray_img,resize_shape, interpolation=cv.INTER_AREA)
    return p_img

def start_learning(num_of_img):
    print('Learning......')
    nn.update = True
    for i in range(1,num_of_img+1):
        img = cv.imread('img/training_img_test3/training_img_{}.jpg'.format(i))
        img = preprocess(img)
        en = nn(img.flatten())
        print(en)
    nn.update = False
    print('Learning finshed')

def get_familiarity(img_path,pixels_shift_per_step=10):
    img = cv.imread(img_path)
    img = preprocess(img)
    width_pixels_num = img.shape[1]
    pixel2degree = 360/width_pixels_num
    en_outputs = []
    shifted_img_list = []
    shifted_pixels_steps = list(range(-int(width_pixels_num/2),int(width_pixels_num/2)+1,pixels_shift_per_step))
    look_angle_list = list(range(-180,181,int(pixels_shift_per_step*pixel2degree)))


    index = 0
    for step in shifted_pixels_steps:
        index+=1
        if step>=0:
            left_img = img[0:img.shape[0], 0: step]
            right_img = img[0:img.shape[0], step: width_pixels_num]
            shifted_img = np.concatenate((right_img,left_img),axis=1)
        else:
            left_img = img[0:img.shape[0], 0: width_pixels_num - abs(step)]
            right_img = img[0:img.shape[0], width_pixels_num - abs(step): width_pixels_num]
            shifted_img = np.concatenate((right_img,left_img),axis=1)
        shifted_img_list.append(shifted_img)
        en_value = nn(img.flatten())
        en_outputs.append(en_value)
        cv.imwrite('img/{}_{}_{}.png'.format(index,step,))
           
    print(en_output)
    i = np.argmin(en_output)
    plt.plot(look_angle_list, en_output)
    plt.xlabel("Angle in degree")
    plt.ylabel("En-value")
    plt.show()

    return shifted_img_list[len(shifted_img_list)-i]

start_learning(14)
img = get_familiarity('img/mental_rotation_test/test_img.png')
# img = cv.imread('img/training_img_test/training_img_1.png')
while(True):
    cv.imshow('test img',img)
    k = cv.waitKey(1)
    if k == ord('q'):
        cv.destroyWindow('test img')
        break
