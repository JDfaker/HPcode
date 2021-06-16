import math
import numpy as np

class Kinematic:

    def __init__(self, init_joint):
        self.l1 = 1
        self.l2 = 1
        self.l3 = 1
        self.l4 = 1

        self.init_joint = init_joint

    def foward_kinematic(self, joint):

        #degree to radian
        theta1 = math.radians(joint[0] + self.init_joint[0])
        theta2 = math.radians(joint[1] + self.init_joint[1])
        theta3 = math.radians(joint[2] + self.init_joint[2])
        theta4 = math.radians(joint[3] + self.init_joint[3])

        p = [math.cos(theta1) * (self.l1 + self.l2*math.cos(theta2) + self.l3*math.cos(theta2+theta3) + self.l4*math.cos(theta2+theta3+theta4)),
            math.sin(theta1) * (self.l1 + self.l2*math.cos(theta2) + self.l3*math.cos(theta2+theta3) + self.l4*math.cos(theta2+theta3+theta4)),
            self.l2*math.sin(theta2) + self.l3*math.sin(theta2+theta3) + self.l4*math.sin(theta2+theta3+theta4)]

        # c1 = math.cos(theta1)
        # c2 = math.cos(theta2)
        # c3 = math.cos(theta3)
        # c4 = math.cos(theta4)
        # s1 = math.sin(theta1)
        # s2 = math.sin(theta2)
        # s3 = math.sin(theta3)
        # s4 = math.sin(theta4)

        #Transform matrix
        # T_01 = np.array([
        #     [c1, 0, s1, self.l1*c1],
        #     [s1, 0, -c1, self.l1*s1],
        #     [0, 1, 0, 0],
        #     [0, 0, 0, 1]
        # ])

        # T_12 = np.array([
        #     [c2, -s2, 0, self.l2*c2],
        #     [s2, c2, 0, self.l2*s2],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]
        # ])

        # T_23 = np.array([
        #     [c3, -s3, 0, self.l3*c3],
        #     [s3, c3, 0, self.l3*s3],
        #     [0, 0 , 1, 0],
        #     [0, 0, 0, 1]
        # ])

        # T_34 = np.array([
        #     [c4, -s4, 0, self.l4*c4],
        #     [s4, c4, 0, self.l4*s4],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]
        # ])
        # T_04 = np.dot(np.dot(np.dot(T_01, T_12), T_23), T_34)

        return p

    def inverse_kinematic(self, position):
        pass
    


if __name__ == '__main__':
    k = Kinematic([0,-40,100,-10])
    t,p = k.foward_kinematic([0,0,0,0])
    print(p)