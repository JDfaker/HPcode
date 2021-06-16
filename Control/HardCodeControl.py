import serial
import time
import RobotConfig as RC


#coxa
#femur
#tibia
#tars

class Robot:
    def __init__(self,port):
        self.control_board = serial.Serial(port)

    def set_all_legs_to_defualt(self):
        time.sleep(0.5)
        self.control_board.write('#0 P1500 #1 P1500 #2 P1500 #3 P1500 #4 P1500 #5 P1500 #6 P1500 #7 P1500 #8 P1500 #9 P1500 #10 P1500 #11 P1500 #16 P1500 #17 P1500 #18 P1500 #19 P1500 #20 P1500 #21 P1500 #22 P1500 #23 P1500 #24 P1500 #25 P1500 #26 P1500 #27 P1500 T5000\r'.encode())
        time.sleep(1)

    def return_all_legs_to_default(self):
        
        self.control_board.write(('#{}P1700 T1000\r'.format(RC.rf_femur_pin)).encode())#rf lift
        time.sleep(1)
        self.control_board.write(('#{}P1500 #{}P1500 T1000\r'.format(RC.rf_coxa_pin,RC.rf_femur_pin)).encode())#rf down
        time.sleep(1)

        self.control_board.write(('#{}P1700 T1000\r'.format(RC.rm_femur_pin)).encode())#rm lift
        time.sleep(1)
        self.control_board.write(('#{}P1500 #{}P1500 T1000\r'.format(RC.rm_coxa_pin,RC.rm_femur_pin)).encode())#rm down
        time.sleep(1)

        self.control_board.write(('#{}P1700 T1000\r'.format(RC.rr_femur_pin)).encode())#rr lift
        time.sleep(1)
        self.control_board.write(('#{}P1500 #{}P1500 T1000\r'.format(RC.rr_coxa_pin,RC.rr_femur_pin)).encode())#rr down
        time.sleep(1)

        self.control_board.write(('#{}P1300 T1000\r'.format(RC.lf_femur_pin)).encode())#lf lift
        time.sleep(1)
        self.control_board.write(('#{}P1500 #{}P1500 T1000\r'.format(RC.lf_coxa_pin,RC.lf_femur_pin)).encode())#lf down
        time.sleep(1)

        self.control_board.write(('#{}P1300 T1000\r'.format(RC.lm_femur_pin)).encode())#lm lift
        time.sleep(1)
        self.control_board.write(('#{}P1500 #{}P1500 T1000\r'.format(RC.lm_coxa_pin,RC.lm_femur_pin)).encode())#lm down
        time.sleep(1)

        self.control_board.write(('#{}P1300 T1000\r'.format(RC.lr_femur_pin)).encode())#rf lift
        time.sleep(1)
        self.control_board.write(('#{}P1500 #{}P1500 T1000\r'.format(RC.lr_coxa_pin,RC.lr_femur_pin)).encode())#rf down
        time.sleep(1)


    def tripod_gait(self,num_of_steps):
        for i in range(num_of_steps):
            self.control_board.write('#1P1700 #9P1700 #21P1300 T1000\r'.encode())#rf,rr,lm lift
            time.sleep(1)
            self.control_board.write('#4P1300 #16P1700 #24P1700 T1000\r'.encode())#rm,lf,lr propel
            time.sleep(1)
            self.control_board.write('#0P1700 #8P1700 #20P1300 T1000\r'.encode())#rf,rr,lm forward
            time.sleep(1)
            self.control_board.write('#1P1500 #9P1500 #21P1500 T1000\r'.encode())#rf,rr,lm down
            time.sleep(1)
            self.control_board.write('#5P1700 #17P1300 #25P1300 T1000\r'.encode())#rm,lf,lr lift
            time.sleep(1)
            self.control_board.write('#0P1300 #8P1300 #20P1700 T1000\r'.encode())#rf,rr,lm, propel
            time.sleep(1)
            self.control_board.write('#4P1700 #16P1300 #24P1300 T1000\r'.encode())#rm,lf,lr forward
            time.sleep(1)
            self.control_board.write('#5P1500 #17P1500 #25P1500 T1000\r'.encode())#rm,lf,lr down
            time.sleep(1)
        
        self.return_all_legs_to_default()
    
    def rotation_gait(self, angle):
        assert -45<=angle and angle<=45, 'the rotation angle have to be -45 <= angle <= 45 degree'
        pulse = round(RC.degree2pulse*angle)
        
        self.control_board.write('#1P1700 #9P1700 #21P1300 T1000\r'.encode())#rf,rr,lm lift
        time.sleep(1)
        self.control_board.write('#0P1700 #8P1700 #20P1300 T1000\r'.encode())#rf,rr,lm forward
        time.sleep(1)


    def turn_gait(angle):
        None
    
robot = Robot('COM4')
robot.set_all_legs_to_defualt()
robot.tripod_gait(2)