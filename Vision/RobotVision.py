from insectvision.net.willshawnet import WillshawNet
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


import sys
sys.path.append(r"C:\\Users\\JD\\Desktop\\HPcode\\ricoh-theta-module-master\\")
from thetacv.subsampling.models import SmolkaModel, CustomModel, vec2sph, sph2vec
from thetacv.capture.stream import Frame, Camera
from thetacv.capture.preprocess import OmmatidiaVoronoi, OmmatidiaFeatures, NonePreprocess, Dewarp, Panorama

class Vision:
    def __init__(self):
        self.nn = WillshawNet(nb_channels=1)

        self.num_training_img_captured = 0
        self.num_img_captured = 0
        self.learning_flag = True

        self.training_images_path = 'img/experiment_4/path_learning_img/'
        self.retracing_images_path = 'img/experiment_4/path_retrace_img/'

    def preprocess(sefl,image):
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        p_img = cv.resize(gray_img,(200,10), interpolation=cv.INTER_AREA)
        return p_img
    
    def start_learning(self):
        print('Learning......')
        self.nn.update = True
        for i in range(1,self.num_training_img_captured+1):
            img = cv.imread(self.training_images_path + 'img_{}.jpg'.format(i))
            img = self.preprocess(img)
            en = self.nn(img.flatten())
            print(en)
        self.nn.update = False
        print('Learning finshed')
    
    def capture_image(self):
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        ret, frame = cap.read()
        cv.imwrite('img/path_learning_img/img_{}.jpg'.format(self.num_training_img_captured),frame,cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
        self.num_img_captured += 1
        cap.release()
    
    def get_highest_familiarity(self,img_path,pixels_shift_per_step=5):
        img = cv.imread(img_path)
        img = self.preprocess(img)
        width_pixels_num = img.shape[1]
        en_output = []
        shifted_img_list = []
        shifted_pixels_steps = list(range(-100,101,pixels_shift_per_step))
        look_angle_list = list(range(-180,181,int(pixels_shift_per_step*1.8)))

        for step in shifted_pixels_steps: 
            if step>=0:
                left_img = img[0:img.shape[0], 0: step]
                right_img = img[0:img.shape[0], step: width_pixels_num]
                shifted_img = np.concatenate((right_img,left_img),axis=1)
            else:
                left_img = img[0:img.shape[0], 0: width_pixels_num - abs(step)]
                right_img = img[0:img.shape[0], width_pixels_num - abs(step): width_pixels_num]
                shifted_img = np.concatenate((right_img,left_img),axis=1)
            en_output.append(self.nn(shifted_img.flatten()))
            shifted_img_list.append(shifted_img)
        
        plt.plot(look_angle_list, en_output)
        plt.xlabel("angle")
        plt.ylabel("familiarity")
        plt.show()

        # i = np.argmin(en_output)
        # plt.imshow(shifted_img_list[i],cmap=plt.cm.gray)
    
    def streaming_with_fisheyeView(self):
        cap = cv.VideoCapture(0)
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Display the resulting frame
            cv.imshow('Live', frame)

            k = cv.waitKey(1)
            if k == ord('q'):
                cv.destroyWindow('Live')
                break
            elif k == ord('c'):
                if(self.learning_flag):
                    self.num_training_img_captured += 1
                    cv.imwrite(self.training_images_path + 'img_{}.jpg'.format(self.num_training_img_captured),frame)
                else:
                    self.num_img_captured += 1
                    cv.imwrite(self.retracing_images_path + 'img_{}.jpg'.format(self.num_img_captured),frame)
                print('image saved')
        cap.release()
    
    def streaming_with_humanView(self):
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
                cv.destroyWindow('Live')
                break
            elif k == ord('c'):
                if(self.learning_flag):
                    self.num_training_img_captured += 1
                    cv.imwrite(self.training_images_path + 'img_{}.jpg'.format(self.num_training_img_captured),img)
                else:
                    self.num_img_captured += 1
                    cv.imwrite(self.retracing_images_path + 'img_{}.jpg'.format(self.num_img_captured),img)
                print('image saved')

    def streaming_with_crabView(self):
        # % of the radius of the fisheye image
        perc_radius = 0.9

        # lenght of the radius of the fisheye image in pixels
        radius_fisheye = 160

        # default previewed image size
        img_size = (1024, 512)

        r = perc_radius * radius_fisheye
        # use the ommatidia model of Smolka(2009)
        model = SmolkaModel(radius=r, rebuild=True)
        preprocessor = OmmatidiaVoronoi(model=model, colour_spec=False, \
                                        rebuild_map=True)
        self.stream = Camera(preprocessor=preprocessor)

        while True:
            # get the new frame
            frame = self.stream.next()

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
                cv.destroyWindow('Live')
                break
            elif k == ord('c'):
                if(self.learning_flag):
                    self.num_training_img_captured += 1
                    cv.imwrite(self.training_images_path + 'img_{}.jpg'.format(self.num_training_img_captured),frame)
                else:
                    self.num_img_captured += 1
                    cv.imwrite(self.retracing_images_path + 'img_{}.jpg'.format(self.num_img_captured),frame)
                print('image saved')

RV = Vision()
try:
    while True:
        command = input("'1':streaming with original fisheye view\n'2':streaming with Human view\n'3':streaming with Crab view\n'4':Start learning\n'5':set learning flag\n'0':get_highest_familiarity\n'end': exit programm\n")
        if command == '1':
            RV.streaming_with_fisheyeView()
            print('press \'c\' to capture image')
            print('press \'q\' quit streaming')
        if(command == '2'):
            RV.streaming_with_humanView()
            print('press \'c\' to capture image')
            print('press \'q\' quit streaming')
        if(command == '3'):
            RV.streaming_with_crabView()
            print('press \'c\' to capture image')
            print('press \'q\' quit streaming')
        if command == '4':
            RV.start_learning()
        if command == '5':
            RV.learning_flag = not RV.learning_flag
            print('Is learning?: '+str(RV.learning_flag))
        if command == '0':
            RV.get_highest_familiarity(RV.retracing_images_path + 'img_{}.jpg'.format(RV.num_img_captured))
        if command == 'end':
            break
except KeyboardInterrupt:
    print("End")
    