#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:06:12 2023

@author: sourav
"""

import os
import numpy as np
import time
import rospy
from gazebo_msgs.srv import GetModelState
import matplotlib.pyplot as plt

debug = False

states = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

d_array = np.arange(1, 10.1, 1)
v_array = np.arange(1, 10.9, 0.5)

dataset = {}

y_pos = [0]

for i in d_array:
    E_array = []
    velocity_dataset = {}
    for j in v_array:
        En_path = 0
        for y in y_pos:
            os.system("rosservice call /gazebo/reset_world")
            os.system("python3 energy_model.py " + str(i) + " 0 2")
            time.sleep(1)
            x = states('bebop2','').pose.position.x
            # import pdb; pdb.set_trace()
            while(abs(x)-i > 0.1):
                os.system("rosrun rotors_gazebo waypoint_publisher " + str(i) +" 0 1.5 180 0 __ns:=bebop2")
                x = states('bebop2','').pose.position.x
                # import pdb; pdb.set_trace()
            time.sleep(1)
            os.system("python3 energy_model.py 0 " + str(y) + " " + str(j) + " >> tmp.txt")
            with open("tmp.txt", "r") as tmp:
                lines = tmp.readlines()
                for line in lines:
                    if line.find("Energy burned by motor currents = ") != -1:
                        En = float(line.strip().split()[-2])
                        En_path += En
            os.system("rm tmp.txt")
        E_array.append(En_path/len(y_pos))
    model = np.poly1d(np.polyfit(np.array(v_array),np.array(E_array),5))
    # import pdb; pdb.set_trace()
    if debug == True:
        E_test = []
        v = v_array
        coeff = model.c
        for ii in range(len(v)):
            E_test.append(coeff[0]*v[ii]**5 + coeff[1]*v[ii]**4 + coeff[2]*v[ii]**3 + coeff[3]*v[ii]**2 + coeff[4]*v[ii] + coeff[5])
        plt.subplot(121)
        plt.plot(v,E_test, "r")
        plt.title("Reconstructed")
        
        plt.subplot(122)
        plt.plot(v, E_array, "g")
        plt.title("Experimentally Obtained")
        plt.show()
    
    dataset[i]=  v_array[E_array.index(min(E_array))], np.array(model.c) 


np.save("data/dataset", dataset, allow_pickle=True, fix_imports=True)


