#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:59:58 2023

@author: sourav
"""

import  rospy
import gazebo_msgs.msg as gz
import mav_msgs .msg
import std_msgs.msg
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Twist, Transform,  Quaternion
import tf
import math
import sys
import params as p
import numpy as np
import os


class Trajectory():
    def __init__(self):
        self.msg = MultiDOFJointTrajectory()
        header = std_msgs.msg.Header()
        header.frame_id = 'frame'
        self.msg.header = header
        self.msg.joint_names.append('base_link')
        self.velocities = Twist()
        self.velocities.linear.x = float(sys.argv[3])
        self.accelerations = Twist()
        self.desired_yaw_degree = 180
        self.quaternion = tf.transformations.quaternion_from_euler(0, 0, math.radians(self.desired_yaw_degree))
        self.t_traj=0
        
 
    def update(self):
        global y
        global z
        global x_drone
        self.t_traj = abs(x_drone - x_target)/self.velocities.linear.x
        rospy.loginfo_once("trajectory time = " + str(self.t_traj))     
        if (x_target < x_drone):
            self.velocities.linear.x = - float(sys.argv[3])
        self.msg.header.stamp = rospy.Time.now()
        transforms = Transform()
        transforms.translation.x = x_drone
        transforms.translation.y = float(sys.argv[2])
        transforms.translation.z = 1.5
        transforms.rotation = Quaternion(self.quaternion[0], self.quaternion[1], self.quaternion[2], self.quaternion[3])
        point = MultiDOFJointTrajectoryPoint()
        point.transforms = [transforms]
        point.velocities = [self.velocities]
        self.msg.points.append(point)

        return self.msg
    
    
            
def callback(data):
    global traj
    global x_drone
    global Energy
    global command_publisher
    global x_pos
    x_drone = data.pose[1].position.x
    x_pos.append(x_drone)
    traj = Trajectory()
    command_publisher.publish(traj.update())
    if abs(x_drone - x_target) < 0.1:
        # os.system("rosrun rotors_gazebo waypoint_publisher 0 0 1.5 180 0 __ns:=bebop2")
        Flight_Energy = sum(Energy)
        rospy.loginfo("Drone has reached destination! Exiting ...")
        print("Energy burned by motor currents = ", Flight_Energy, "J")
        # print("Depth traversed =", abs(x_pos[-1] - x_pos[0]), "m")
        # import pdb; pdb.set_trace()
        rospy.signal_shutdown("exit")

        
def speed_logger(data):
    global speeds
    global t_prev, t_curr, dt
    t_prev = t_curr
    t_curr = data.header.stamp.secs+(data.header.stamp.nsecs/1e9)
    dt = t_curr - t_prev 
    w = data.angular_velocities
    speeds.append([w])
    if len(speeds) > 2:       
        alpha = (w - np.array(speeds[-2][0]))/dt
        energy = 0
        for i in range(4):
            energy += (p.c1 + p.c2 * w[i] + p.c3 * w[i]**2 + p.c4 * w[i]**3 + p.c5 * w[i]**4  + p.c6 * alpha[i]**2)*dt
        # print("Instantaneous Power = ",  str(energy/dt) , "W")
        Energy.append(energy)
    
    
def fly_drone():
    rospy.init_node('fly_drone', anonymous=True)
    rospy.Subscriber('/gazebo/model_states/', gz.ModelStates, callback, tcp_nodelay=True)
    rospy.Subscriber('/bebop2/command/motor_speed', mav_msgs.msg.Actuators, speed_logger, tcp_nodelay=True)     
    rospy.spin()

if __name__ == '__main__':
    try:
        t_prev = t_curr = 0
        traj = Trajectory()
        x_target = float(sys.argv[1])
        global y
        y = float(sys.argv[2])
        z = 1.5
        speeds = []
        Energy = []
        x_pos = []
        command_publisher = rospy.Publisher('/bebop2/command/trajectory', MultiDOFJointTrajectory, queue_size=10)
        fly_drone()
    except rospy.ROSInterruptException:
        pass
