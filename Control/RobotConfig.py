import math

#initial angle of each joints
coxa_init = 0
femur_init = 0
tibia_init = 90
tars_init = 0

#convert degree to pulse
degree2pulse = 500/90

#leg length in mm
coxa_length = None
femur_length = None
tibia_length = None
tars_length = None

#SSC pin
rf_coxa_pin = 0
rf_femur_pin = 1
rf_tibia_pin = 2
rf_tars_pin = 3

rm_coxa_pin = 4
rm_femur_pin = 5
rm_tibia_pin = 6
rm_tars_pin = 7

rr_coxa_pin = 8
rr_femur_pin = 9
rr_tibia_pin = 10
rr_tars_pin = 11

lf_coxa_pin = 16
lf_femur_pin = 17
lf_tibia_pin = 18
lf_tars_pin = 19

lm_coxa_pin = 20
lm_femur_pin = 21
lm_tibia_pin = 22
lm_tars_pin = 23

lr_coxa_pin = 24
lr_femur_pin = 25
lr_tibia_pin = 26
lr_tars_pin = 27

#AEP
aep = [math.sqrt(3), -1, -2]
#PEP
pep = [math.sqrt(3), 1, -2]
#
aep_pep_middle = [1.5, 0, -1.5]
#default_position
default_p = [2,0,-2]

#initial angle
#transfomation matrix