import math
import time
from RobotInterface import RobotInterface

class Control(RobotInterface):
    def __init__(self,port):
        super(Control,self).__init__(port)

    def tripod_gait(self,steps):
        for i in range(steps):
            self.rotate_to([('rf',1,30),('rr',1,30),('lm',1,-30),
                            ('rf',0,30),('rr',0,30),('lm',0,-30)],T=500)
            time.sleep(0.1)

            self.rotate_to([('lf',0,30),('lr',0,30),('rm',0,-30)],T=1000)
            time.sleep(1)

            self.rotate_to([('rf',1,0),('rr',1,0),('lm',1,0),
                            ('rf',0,30),('rr',0,30),('lm',0,-30)],T=500)

            time.sleep(1)
            self.rotate_to([('lf',0,0),('lr',0,0),('rm',0,0),
                            ('lf',1,-30),('lr',1,-30),('rm',1,30)],T=500)
            time.sleep(0.1)

            self.rotate_to([('rf',0,-30),('rr',0,-30),('lm',0,30)],T=1000)
            time.sleep(1)

            self.rotate_to([('lf',0,-30),('lr',0,-30),('rm',0,30),
                            ('lf',1,0),('lr',1,0),('rm',1,0)],T=500)
            time.sleep(1)
        
        self.return_leg_to_default()

    def rotation_gait(self):
        pass
    def turn_gait(self):
        pass

    def return_leg_to_default(self):
        self.rotate_to([('rf',1,30),('rr',1,30),('lm',1,-30)],T=1000)
        time.sleep(2)
        self.rotate_to([('rf',1,0),('rr',1,0),('lm',1,0),
                            ('rf',0,0),('rr',0,0),('lm',0,0)],T=1000)
        time.sleep(2)
        self.rotate_to([('lf',1,-30),('lr',1,-30),('rm',1,30)],T=1000)
        time.sleep(2)
        self.rotate_to([('lf',0,0),('lr',0,0),('rm',0,0),
                            ('lf',1,0),('lr',1,0),('rm',1,0)],T=1000)
    
    def scan(self):
        pass

robot = Control('COM4')
robot.set_all_legs_to_defualt()
time.sleep(2)
try:
    while True:
        command = input('1:tripod_gait\n2:return legs to default\n')
        if command == '1':
            robot.tripod_gait(20)
        if command == '2':
            robot.return_leg_to_default()
except KeyboardInterrupt:
    print("End")
robot.close_port()
