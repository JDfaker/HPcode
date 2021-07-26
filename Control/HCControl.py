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
            self.rotate_to([('rf',0,45),('rr',0,45),('lm',0,-45)],T=500)
            
            #lf lr rm propel
            self.rotate_to([('lf',0,45),('lr',0,45),('rm',0,-45)],T=1000)
            time.sleep(1)

            #put down rf rr lm
            self.rotate_to([('rf',1,0),('rr',1,0),('lm',1,0)],T=500)
            time.sleep(1)

            #lift lf lr rm
            self.rotate_to([('lf',1,-45),('lr',1,-45),('rm',1,45)],T=500)
            time.sleep(0.3)
            #forward lf lr rm
            self.rotate_to([('lf',0,-45),('lr',0,-45),('rm',0,45)],T=500)

            self.rotate_to([('rf',0,-45),('rr',0,-45),('lm',0,45)],T=1000)
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


    def turn_gait(self):
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
    
    def scan(self):
        pass

robot = Control('COM4')
robot.set_all_legs_to_defualt()
time.sleep(1)
robot.set_offset()
time.sleep(1)
try:
    while True:
        command = input('1:tripod_gait\n2:rotation_gait\n3:return legs to default\n')
        if command == '1':
            robot.tripod_gait(10)
        elif command == '2':
            robot.rotation_gait(45)
        elif command == '3':
            robot.return_leg_to_default()
except KeyboardInterrupt:
    print("End")
robot.close_port()
