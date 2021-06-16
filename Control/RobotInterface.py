import serial
import math
import time

Legs = ['rf','rm','rr','lf','lm','lr']
Joints = [0,1,2,3]

Joints_pins = {'rf':['0','1','2','3'],
            'rm':['4','5','6','7'],
            'rr':['8','9','10','11'],
            'lf':['16','17','18','19'],
            'lm':['20','21','22','23'],
            'lr':['24','25','26','27']}

Degree2pulse = 500/90

Offset = {'0':0, '1':0, '2':0, '3':0,
          '4':0, '5':0, '6':0, '7':0,
          '8':0, '9':0, '10':0, '11':0,
          '16':0, '17':0, '18':0, '19':0,
          '20':0, '21':0, '22':0, '23':0,
          '24':0, '25':0, '26':0, '27':0,}

class RobotInterface:
    def __init__(self,port):
        self.port = port
        self.control_board = serial.Serial(self.port)

        self.joints_pos = {'0':0, '1':0, '2':0, '3':0,
                           '4':0, '5':0, '6':0, '7':0,
                           '8':0, '9':0, '10':0, '11':0,
                           '16':0, '17':0, '18':0, '19':0,
                           '20':0, '21':0, '22':0, '23':0,
                           '24':0, '25':0, '26':0, '27':0,}

    def get_joint_pulse(self,leg,joint):
        pin = Joints_pins[leg][joint]
        print(self.joints_pos[pin])
        return self.joints_pos[pin]

    def rotate_to(self,movements,T=2000,S=None):
        #p : (leg,joint,angle) angle in degree
        assert (type(movements) == list) or (type(movements) == tuple), 'Input should be tuple (leg,joint,angle) or list [(leg,joint,angle)]'

        if (type(movements) == tuple):
            assert (movements[0] in Legs), "the first argument of the tuple should be in ['rf','rm','rr','lf','lm','lr']"
            assert (movements[1] in Joints), "the second argument of the tuple should be in [0,1,2,3]"

            pin = Joints_pins[movements[0]][movements[1]]
            pulse = round(Degree2pulse*movements[2]) + 1500
            
            if S != None:
                self.control_board.write(('#{} P{} T{} S{}\r'.format(pin,pulse,T,S)).encode())
            else:
                self.control_board.write(('#{} P{} T{}\r'.format(pin,pulse,T)).encode())

            self.joints_pos[pin] = movements[2]

        else:
            command = ''
            for move in movements:
                pin = Joints_pins[move[0]][move[1]]
                pulse = round(Degree2pulse*move[2]) + 1500
                command = command + '#' + pin + ' ' + 'P' + str(pulse) + ' '
                self.joints_pos[pin] = move[2]

            if S != None:
                command  = command + 'T{} S{}\r'.format(T,S)
            else:
                command  = command + 'T{}\r'.format(T)
            
            self.control_board.write(command.encode())
    
    def set_offset(self):
        command = ''
        for key,value in Offset:
            command = command + '#{} PO{}'.format(key,value)
        
        command = command + '\r'
        self.control_board.write(command.encode())


    def tune_offset(self):
        try:
            leg = input('Choose leg(rf,rm,rr,lf,lm,lr): ')
            joint = int(input('Choose joint(0,1,2,3): '))
            pin = Joints_pins[leg][joint]
            while True:
                offset = int(input('Offset(-100<=offset<=100): '))
                assert offset<=100 and offset>=-100, 'offset must be <=100 and >=-100'
                self.control_board.write(('#{} PO{}\r'.format(pin, offset)).encode())
        except KeyboardInterrupt:
            print("Tune End")

    def set_all_legs_to_defualt(self):
        self.control_board.write('#0 P1500 #1 P1500 #2 P1500 #3 P1500 #4 P1500 #5 P1500 #6 P1500 #7 P1500 #8 P1500 #9 P1500 #10 P1500 #11 P1500 #16 P1500 #17 P1500 #18 P1500 #19 P1500 #20 P1500 #21 P1500 #22 P1500 #23 P1500 #24 P1500 #25 P1500 #26 P1500 #27 P1500 T1000\r'.encode())
        for key in self.joints_pos:
            self.joints_pos[key] = 1500
    
    def close_port(self):
        self.control_board.close()

        