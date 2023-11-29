#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:59:58 2023

@author: sourav
"""

import  rospy
import gazebo_msgs.msg as gz
from gazebo_msgs.srv import GetModelState
import std_msgs.msg
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Twist, Transform, Point, Quaternion
import tf
import math
import time
from tensorflow import keras
import numpy as np
from physics_guided_nn import PgNN


def callback(data):
    global x_target, y_ring_now, y_target, x_drone, depth, vel_x_drone, transforms, model
    desired_yaw_degree = 180
    z_target = 1.5
    x_target = data.pose[2].position.x - 1
    y_ring_now = data.pose[2].position.y    # use events 
    # print("y_now = " , y_ring_now)
    x_drone = data.pose[3].position.x
    depth = abs(x_target - x_drone)
    vel_x_drone = data.twist[3].linear.x
    quaternion = tf.transformations.quaternion_from_euler(0, 0, math.radians(desired_yaw_degree))
    
    inp = np.array([depth])
 
    #Calculate y_target from y_now
    ring_vel = 0.8 #m/s
    v_opt = model.predict(inp) #NN with depth
    t_traj = inp/v_opt[0][0]
    states = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    y1 = states('moving_ring','').pose.position.y   #same as y_ring_now
    time.sleep(0.1)
    y2 =  states('moving_ring','').pose.position.y
    
    
    d1 = ring_vel * t_traj
    
    if y2 > y1:  #moving right
        if y2 > 0:  
            d2 = 2 - y2
        elif y2 < 0:
            d2 = 2 + abs(y2)
        if d2:
            if d1 > d2:
                y_target = 2 - d1 + d2
            elif d1 < d2:
                y_target = y2 + d1
            
    elif y2 < y1: # moving left
        if y2 < 0:
            d2 = abs(-2 - y2)
        elif y2 > 0:
            d2 = 2 + y2
        if d2:
            if d1 > d2:
                y_target = -2 + d1 -d2
            elif d1 < d2:
                y_target = y2 - d1
    
    
    
    transforms = Transform(translation=Point(x_target, y_target, z_target), rotation=Quaternion(quaternion[0],quaternion[1],quaternion[2],quaternion[3]))
    traj = MultiDOFJointTrajectory()

    header = std_msgs.msg.Header()
    header.stamp = rospy.Time()
    header.frame_id = 'frame'
    traj.joint_names.append('base_link')
    traj.header=header
    
    velocities = Twist()
    # velocities.linear.x = v_opt
    accelerations=Twist()
    point = MultiDOFJointTrajectoryPoint([transforms],[velocities],[accelerations],rospy.Time(t_traj))
    traj.points.append(point)
    command_publisher = rospy.Publisher('/bebop2/command/trajectory', MultiDOFJointTrajectory, queue_size=10)
    command_publisher.publish(traj)

    # import pdb; pdb.set_trace()

    
def trajectory_learner():
    rospy.init_node('trajectory_learner', anonymous=True)

    
    while not rospy.is_shutdown():

        rospy.Subscriber('/gazebo/model_states/', gz.ModelStates, callback)
        rospy.spin()

if __name__ == '__main__':
    try:
        model = PgNN()
        model.load_weights('model')
        # input("Model loaded! Press Enter to start Trajectory Learner")
        trajectory_learner()
    except rospy.ROSInterruptException:
        pass
