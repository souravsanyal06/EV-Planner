#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:50:12 2023

@author: sourav
"""

import  rospy
import gazebo_msgs.msg as gz
from gazebo_msgs.srv import GetModelState
import std_msgs.msg
import matplotlib.pyplot as plt
import numpy as np

X = []
Y = []
Z = []


# if __name__ == '__main__':
    
    # try:
    #     while True:
    #         states = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    #         x = states('bebop2','').pose.position.x
    #         y = states('bebop2','').pose.position.y
    #         z = states('bebop2','').pose.position.z
    #         X.append(x)
    #         Y.append(y)
    #         Z.append(z)
    # except KeyboardInterrupt:
    #     plt.subplot(1,3,1)
    #     plt.plot(X)
    #     plt.plot([0]*len(X))
    #     plt.plot([1]*len(X))
    #     plt.plot([2]*len(X))
    #     plt.title('X')
    #     plt.xlabel('time (ms)')
    #     plt.subplot(1,3,2)
    #     plt.plot(Y)
    #     plt.plot([0]*len(Y))
    #     plt.plot([1]*len(Y))
    #     plt.plot([2]*len(Y))
    #     plt.title('Y')
    #     plt.xlabel('time (ms)')
    #     plt.subplot(1,3,3)
    #     plt.plot(Z)
    #     plt.plot([0]*len(Z))
    #     plt.plot([1]*len(Z))
    #     plt.plot([2]*len(Z))
    #     plt.title('Z')
    #     plt.xlabel('time (ms)')
    #     plt.show()
    
    
    
X = np.load('x.npy')
Y = np.load('y.npy')
Z = np.load('z.npy')    

# import pdb; pdb.set_trace()

def check_near_0_1_2(val):
    t0 = np.abs(val)
    t1 = np.abs(val-1)
    t2 = np.abs(val-2)
    
    v = min(t0,t1,t2)
    
    if v == t0:
        return val
    if v == t1:
        return abs(val-1)
    if v == t2:
        return abs(val-2)
    
e_X =[]

for i in X:
    e_X.append(check_near_0_1_2(i))
    
e_Y =[]

for i in Y:
    e_Y.append(check_near_0_1_2(i))
    
e_Z =[]

for i in Z:
    e_Z.append(check_near_0_1_2(i))    
    




# import pdb; pdb.set_trace()

x = np.asanyarray(X)
y = np.asanyarray(Y)
z = np.asanyarray(Z)
m_x = x.mean(axis=0)
m_y = y.mean(axis=0)
m_z = z.mean(axis=0)
sd_x = x.std(axis=0, ddof=0)
sd_y = y.std(axis=0, ddof=0)
sd_z = z.std(axis=0, ddof=0)

snr_x = 10*np.log10(np.where(sd_x == 0, 0, m_x/sd_x))
snr_y = 10*np.log10(np.where(sd_y == 0, 0, m_y/sd_y))
snr_z = 10*np.log10(np.where(sd_z == 0, 0, m_z/sd_z))


e_x = np.asanyarray(e_X)
e_y = np.asanyarray(e_Y)
e_z = np.asanyarray(e_Z)
m_ex = e_x.mean(axis=0)
m_ey = e_y.mean(axis=0)
m_ez = e_z.mean(axis=0)
e_sd_x = e_x.std(axis=0, ddof=0)
e_sd_y = e_y.std(axis=0, ddof=0)
e_sd_z = e_z.std(axis=0, ddof=0)

snr_ex = abs(10*np.log10(np.where(e_sd_x == 0, 0, m_ex/e_sd_x)))
snr_ey = abs(10*np.log10(np.where(e_sd_y == 0, 0, m_ey/e_sd_y)))
snr_ez = abs(10*np.log10(np.where(e_sd_z == 0, 0, m_ez/e_sd_z)))


# print("Signal to Noise Ratio for position vectors")
# print("SNR for x = " , snr_x)
# print("SNR for y = " , snr_y)
# print("SNR for z = " , snr_z)

print("Signal to Noise Ratio for error vectors")
print("SNR for e_x = " , snr_ex)
print("SNR for e_y = " , snr_ey)
print("SNR for e_z = " , snr_ez)

plt.subplot(2,3,1)
plt.plot(X)
plt.plot([0]*len(X))
plt.plot([1]*len(X))
plt.plot([2]*len(X))
plt.title('X')
plt.xlabel('time (ms)')
plt.subplot(2,3,2)
plt.plot(Y)
plt.plot([0]*len(Y))
plt.plot([1]*len(Y))
plt.plot([2]*len(Y))
plt.title('Y')
plt.xlabel('time (ms)')
plt.subplot(2,3,3)
plt.plot(Z)
plt.plot([0]*len(Z))
plt.plot([1]*len(Z))
plt.plot([2]*len(Z))
plt.title('Z')
plt.xlabel('time (ms)')
plt.subplot(2,3,4)
plt.plot(e_X)
plt.title('X error')
plt.xlabel('time (ms)')
plt.subplot(2,3,5)
plt.plot(e_Y)
plt.title('Y error')
plt.xlabel('time (ms)')
plt.subplot(2,3,6)
plt.plot(e_Z)
plt.title('Z error')
plt.xlabel('time (ms)')
plt.show()
    


