import math
import time
from RobotInterface import RobotInterface

class Control(RobotInterface):
    def __init__(self,port):
        super(Control,self).__init__(port)

    def tripod_gait(self,steps):
        for i in range(steps):
            #lift rf rr lm
            self.rotate_to([('rf',1,45),('rr',1,45),('lm',1,-45)],T=500)
            time.sleep(0.3)
            #forward rf rr lm
            self.rotate_to([('rf',0,30),('rr',0,30),('lm',0,-30)],T=500)
            
            #lf lr rm propel
            self.rotate_to([('lf',0,30),('lr',0,30),('rm',0,-30)],T=1000)
            time.sleep(1)

            #put down rf rr lm
            self.rotate_to([('rf',1,0),('rr',1,0),('lm',1,0)],T=500)
            time.sleep(1)

            #lift lf lr rm
            self.rotate_to([('lf',1,-45),('lr',1,-45),('rm',1,45)],T=500)
            time.sleep(0.3)
            #forward lf lr rm
            self.rotate_to([('lf',0,-30),('lr',0,-30),('rm',0,30)],T=500)

            #propel rf rr lm
            self.rotate_to([('rf',0,-30),('rr',0,-30),('lm',0,30)],T=1000)
            time.sleep(1)

            #put down lf lr rm
            self.rotate_to([('lf',1,0),('lr',1,0),('rm',1,0)],T=500)
            time.sleep(1)
        
        self.return_leg_to_default()

    def rotation_gait(self,rotation_angle):
        self.rotate_to([('rf',0,rotation_angle),('rm',0,rotation_angle),('rr',0,rotation_angle),
                        ('lf',0,rotation_angle),('lm',0,rotation_angle),('lr',0,rotation_angle)],T=1000)
        time.sleep(2)
        self.return_leg_to_default()
    
    def rotation_gait2(self,rotation_angle):
        #lift rf rr lm
        self.rotate_to([('rf',1,45),('rr',1,45),('lm',1,-45)],T=500)
        time.sleep(0.3)
        #forward rf rr lm
        self.rotate_to([('rf',0,30),('rr',0,30),('lm',0,30)],T=500)
        time.sleep(2)
        self.return_leg_to_default()


    def right_turn_gait1(self,steps):
        for i in range(steps):
            #lift rf rr lm
            self.rotate_to([('rf',1,45),('rr',1,45),('lm',1,-45)],T=500)
            time.sleep(0.3)
            #forward rf rr lm
            self.rotate_to([('rf',0,15),('rr',0,15),('lm',0,-20)],T=500)
            
            #lf lr rm propel
            self.rotate_to([('lf',0,30),('lr',0,30),('rm',0,-30)],T=1000)
            time.sleep(1)

            #put down rf rr lm
            self.rotate_to([('rf',1,0),('rr',1,0),('lm',1,0)],T=500)
            time.sleep(1)

            #lift lf lr rm
            self.rotate_to([('lf',1,-45),('lr',1,-45),('rm',1,45)],T=500)
            time.sleep(0.3)
            #forward lf lr rm
            self.rotate_to([('lf',0,-30),('lr',0,-30),('rm',0,30)],T=500)

            #propel rf rr lm
            self.rotate_to([('rf',0,-30),('rr',0,-30),('lm',0,30)],T=1000)
            time.sleep(1)

            #put down lf lr rm
            self.rotate_to([('lf',1,0),('lr',1,0),('rm',1,0)],T=500)
            time.sleep(1)


    def right_turn_gait2(self):
        pass
    def right_turn_gait3(self):
        pass
    def left_turn_gait1(self,steps):
        for i in range(steps):
            #lift rf rr lm
            self.rotate_to([('rf',1,45),('rr',1,45),('lm',1,-45)],T=500)
            time.sleep(0.3)
            #forward rf rr lm
            self.rotate_to([('rf',0,15),('rr',0,15),('lm',0,-20)],T=500)
            
            #lf lr rm propel
            self.rotate_to([('lf',0,30),('lr',0,30),('rm',0,-30)],T=1000)
            time.sleep(1)

            #put down rf rr lm
            self.rotate_to([('rf',1,0),('rr',1,0),('lm',1,0)],T=500)
            time.sleep(1)

            #lift lf lr rm
            self.rotate_to([('lf',1,-45),('lr',1,-45),('rm',1,45)],T=500)
            time.sleep(0.3)
            #forward lf lr rm
            self.rotate_to([('lf',0,-30),('lr',0,-30),('rm',0,30)],T=500)

            #propel rf rr lm
            self.rotate_to([('rf',0,-30),('rr',0,-30),('lm',0,30)],T=1000)
            time.sleep(1)

            #put down lf lr rm
            self.rotate_to([('lf',1,0),('lr',1,0),('rm',1,0)],T=500)
            time.sleep(1)
    def left_turn_gait2(self):
        pass
    def left_turn_gait3(self):
        pass

    def return_leg_to_default(self):
        self.rotate_to([('rf',1,45),('rr',1,45),('lm',1,-45)],T=1000)
        time.sleep(1.5)
        self.rotate_to([('rf',0,0),('rr',0,0),('lm',0,0)],T=1000)
        time.sleep(0.3)
        self.rotate_to([('rf',1,0),('rr',1,0),('lm',1,0)],T=1000)

        time.sleep(1)
        self.rotate_to([('lf',1,-45),('lr',1,-45),('rm',1,45)],T=1000)
        time.sleep(1.5)
        self.rotate_to([('lf',0,0),('lr',0,0),('rm',0,0)],T=1000)
        time.sleep(0.3)
        self.rotate_to([('lf',1,0),('lr',1,0),('rm',1,0)],T=1000)
    

robot = Control('COM3')
robot.set_all_legs_to_defualt()
time.sleep(1)
robot.set_offset()
time.sleep(1)
try:
    while True:
        command = input('1:tripod_gait\n2:right_turn_gait\n3:left_turn_gait\n0:return legs to default\n9:tune offset\n')
        if command == '1':
            num_steps = input('Enter the number of steps: ')
            robot.tripod_gait(int(num_steps))
        elif command == '2':
            num_steps = input('Enter the number of steps: ')
            robot.right_turn_gait1(int(num_steps))
        elif command == '3':
            num_steps = input('Enter the number of steps: ')
            robot.right_turn_gait1(int(num_steps))
        elif command == '0':
            robot.return_leg_to_default()
        elif command == '9':
            robot.tune_offset()
except KeyboardInterrupt:
    print("End")
robot.close_port()
