import serial
from RobotInterface import RobotInterface
import time

RI = RobotInterface('COM4')
RI.set_all_legs_to_defualt()
