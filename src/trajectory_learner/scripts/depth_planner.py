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
import numpy as np
from physics_guided_nn import PgNN
from dvs_msgs.msg import EventArray
import snntorch as snn
import torch
import torch.nn as nn
import cv2
import sensor_msgs.msg as sms
import mav_msgs.msg
import utils.params as p
import sys

from rospy.numpy_msg import numpy_msg

import imutils


############################################ PERCEPTION ####################################################################

def mask_input_logical(input_array, speedy_array, neighborhood_size=1):
   ''' 
   Function to get back inputs within the neighborhood of an output spike.
   '''
   kernel = np.ones((neighborhood_size,neighborhood_size),np.uint8)
   # dilated_speedy_img = cv2.dilate(np.array(speedy_array), kernel, iterations = 1)
   closing = cv2.dilate(np.array(speedy_array), kernel, iterations = 3)
   # closing = cv2.morphologyEx(dilated_speedy_img, cv2.MORPH_CLOSE, kernel)
   # closing = cv2.morphologyEx(np.array(speedy_array), cv2.MORPH_OPEN, kernel, iterations=3)
   # masked_input = np.array(~np.logical_and(input_array,closing))*input_array
   mask=np.array(input_array.shape, dtype=bool)
   mask=~(closing>0)
   # masked_input = np.array(~closing)*input_array
   masked_input = input_array
   masked_input[mask==False] = 0
   return masked_input




def find_marker(image):
   # convert the image to grayscale, blur it, and detect edges
   #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   gray = cv2.GaussianBlur(image, (5, 5), 0)
   # Find Canny edges
   edged = cv2.Canny(gray, 35, 125)
   # find the contours in the edged image and keep the largest one;
   # we'll assume that this is our piece of paper in the image
   cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   cnts = imutils.grab_contours(cnts)
   c = max(cnts, key = cv2.contourArea)
   # compute the bounding box of the of the paper region and return it
   return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
   # compute and return the distance from the maker to the camera
   return (knownWidth * focalLength) / perWidth
     
   
def depthLogger(data):
   global depth_predicted
   global first
   global focalLength
   global KNOWN_WIDTH
   global marker
   global cy_m
   i = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)

   i = cv2.normalize(i, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
   
   states = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
   x_ring = states('moving_ring','').pose.position.x
   x_drone = states('bebop2','').pose.position.x

   if first:
      KNOWN_DISTANCE = abs(x_drone - x_ring) * 39.37  
      # KNOWN_DISTANCE = 3 * 39.37 # convert distance of drone from ring to inches, also check the depth_camera.gazebo
      KNOWN_WIDTH =10.0#check again
      marker = find_marker(i)
      focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
      first=False


   marker = find_marker(i)
   box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
   box = np.int0(box)
   # import pdb; pdb.set_trace()
   # print([box])
   cv2.drawContours(i, [box], -1, (255, 255, 0), 2)

   inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
   # print("%.2f m" % ((inches / 12)*0.3048))

   find_center=True
   if find_center:
      Ximage_x, Ximage_y = np.where(i > 0)
      X = np.zeros((len(Ximage_x), 2))
      X[:,0], X[:,1] = Ximage_x, Ximage_y
      my_image = np.zeros((i.shape))
      for x, y in X:
          my_image[int(x),int(y)] = 255
      
      
      x0 = box[1][1]
      x1 = box[-1][1]
      

      y0 = box[0][0]
      y1 = box[1][0]

      cv2.rectangle(my_image, (y0, x0), (y1, x1), (255, 255, 0), 1)
      cx = x0+(x1-x0)//2
      cy = y0+(y1-y0)//2   #y_star
      cv2.circle(my_image, (cy, cx), 5, (255, 255, 0), 1)#, cv2.LINE_AA)
      
      cy_m = cy * ((4)/640) - 2   #y_star
      
      # cv2.imshow('depth_frame_box_center', my_image)
      # cv2.waitKey(1)  
   
   
   
   ################################################### PLANNING ############################################################
   
def callback(data):
    global x_target, y_ring_now, y_target, x_drone, depth, vel_x_drone, transforms, model, t_prev, cy_m, depth_predicted, shift
    desired_yaw_degree = 180
    
    x_target = data.pose[2].position.x - 1
    z_target = 1.5
          
    y_ring_now = cy_m
    vel_x_drone = data.twist[3].linear.x
    quaternion = tf.transformations.quaternion_from_euler(0, 0, math.radians(desired_yaw_degree))
    # import pdb; pdb.set_trace()
    
    while(not depth_predicted):
        continue
    
    inp = np.array([depth_predicted])
 
    #Calculate y_target from y_now
    ring_vel = 0.8 #m/s
    v_opt = model.predict(inp) #NN with depth
    t_traj = inp/v_opt[0][0]
    states = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    y1 = states('moving_ring','').pose.position.y  
    time.sleep(0.1)
    y2 = states('moving_ring','').pose.position.y
    d1 = ring_vel * t_traj
    
    if y2 > y1:  #moving right
        if y2 > 0:  
            d2 = 2 - y2
        elif y2 < 0:
            d2 = 2 + abs(y2)
        if d2:
            if d1 > d2:
                y_target = 2 - d1 + d2 + int(shift)
            elif d1 < d2:
                y_target = y2 + d1 + int(shift)
            
    elif y2 < y1: # moving left
        if y2 < 0:
            d2 = abs(-2 - y2)
        elif y2 > 0:
            d2 = 2 + y2
        if d2:
            if d1 > d2:
                y_target = -2 + d1 -d2 + int(shift)
            elif d1 < d2:
                y_target = y2 - d1 + int(shift)
        
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
    
    states = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    x_drone = states('bebop2','').pose.position.x
    
   
    if abs(x_drone - x_target) < 0.1:
        Flight_Energy = sum(Energy)
        rospy.loginfo("Drone has reached destination! Exiting ...")
        print("Energy burned by motor currents = ", Flight_Energy, "J")
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

    
def ev_planner():
    rospy.init_node('ev_planner', anonymous=True)
    while not rospy.is_shutdown():
        rospy.Subscriber('/gazebo/model_states/', gz.ModelStates, callback)
        rospy.spin()

if __name__ == '__main__':
    try:
        model = PgNN()
        model.load_weights('model')
        eventFrame=np.zeros((480, 640), dtype=np.uint8)
        t_prev = t_curr = 0
        Energy = []
        speeds = []
        first = True
        flag = False
        depth_predicted = 3
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=True).to('cpu')
        # Set the weights manually
        conv1.weight = torch.nn.Parameter(torch.tensor(torch.ones_like(conv1.weight)*0.1))#0.15
        with torch.no_grad():
           conv1.weight[0, 0, 1, 1] = 0.15#0.2
        # spiking neuron parameters -- LEAK FACTOR = 3
        snn1_lowerbeta = snn.Leaky(beta=0.1, threshold=1.75, reset_mechanism="subtract")#modify beta 0.3
        mem_dir_lowerbeta = snn1_lowerbeta.init_leaky()
        # states = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        # shift = states('bebop2','').pose.position.y
        shift = 0

    
        rospy.Subscriber("/bebop2/depth_camera/depth/image_raw", numpy_msg(sms.Image), depthLogger)
        rospy.sleep(1)
        rospy.Subscriber('/bebop2/command/motor_speed', mav_msgs.msg.Actuators, speed_logger, tcp_nodelay=True)

        ev_planner()
    except rospy.ROSInterruptException:
        pass
