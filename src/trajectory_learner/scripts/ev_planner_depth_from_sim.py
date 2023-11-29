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

def dotie():
   #print(l)
   global img
   global flag
   global conv1, snn1_lowerbeta, mem_dir_lowerbeta
   global cy_m
   if flag:
      flag=False
      data_tensor = torch.tensor(img)

      #visualize event frames
      visual_frame = np.array(data_tensor)
      visual_frame = ((visual_frame - visual_frame.min()) * (1/(visual_frame.max() - visual_frame.min()) * 255)).astype('uint8')
      # cv2.imshow('raw_event_frame', visual_frame)
      # cv2.waitKey(1)
      
      #convert to binary image 255 or 0
      Ximage_x, Ximage_y = np.where(visual_frame > 0)
      X = np.zeros((len(Ximage_x), 2))
      X[:,0], X[:,1] = Ximage_x, Ximage_y
      my_image = np.zeros((visual_frame.shape))
      for x, y in X:
         my_image[int(x),int(y)] = 255



      if not my_image.any():
         return
      x0=Ximage_x.min()
      y0=Ximage_y.min()
      x1=Ximage_x.max()
      y1=Ximage_y.max()
      cv2.rectangle(my_image, (y0, x0), (y1, x1), (255, 255, 0), 1)#, cv2.LINE_AA)
      vanish=False
      show_always = ~vanish
      
      # states = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
      # y_D = states('bebop2','').pose.position.y

      if vanish:
         if (Ximage_x.min()==0):
            biasx=Ximage_x.max()-visual_frame.shape[1]
         elif (Ximage_x.max()==visual_frame.shape[1]):
            biasx=Ximage_x.min()+(visual_frame.shape[1]//2)
         else:
            biasx=x0
         if (Ximage_y.min()==0):
            biasy=Ximage_y.max()-visual_frame.shape[0]
         elif (Ximage_y.max()==visual_frame.shape[0]):
            biasy=Ximage_y.min()+(visual_frame.shape[0]//2)
         else:
            biasy=y0
         cx = biasx+(x1-x0)//2
         cy = biasy+(y1-y0)//2
         if ((0, 0)<=(cy, cx)<=visual_frame.shape):
            cv2.circle(my_image, (cy, cx), 5, (255, 255, 0), 1)#, cv2.LINE_AA)
         cv2.imshow('bounding_box', my_image)
         cv2.waitKey(1)
      if show_always:
         cx = x0+(x1-x0)//2
         cy = y0+(y1-y0)//2
         cy_m = cy * (4/640) - 2   #y_star
         cv2.circle(my_image, (cy, cx), 5, (255, 255, 0), 1)#, cv2.LINE_AA)

         cv2.putText(my_image, "y= %d Pixel" % cy,(my_image.shape[1] - 300, my_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (255, 255, 0), 3)

         cv2.putText(my_image, "x= %d" % cx,(my_image.shape[1] - 600, my_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (255, 255, 0), 3)

         cv2.putText(my_image, "y= %.2fm" % cy_m,(my_image.shape[1] - 400, my_image.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX,2.0, (255, 255, 0), 3)
         # cv2.imshow('without_SNN_bounding_box', my_image)
         cv2.waitKey(1)


      # cinvert input tensor into float type
      inp_img = data_tensor.float()
      # Add two dimensions (batch size and channels)
      inp_img = inp_img[None, None, :]
      #print(inp_img.max(), inp_img.min())
      # Pass it through conv layer
      con_out = conv1(inp_img)
      #print(con_out.max(), con_out.min())
      # sys.exit(0)
      # cv2.imshow('conv_out', con_out[0][0].detach().numpy())
      # cv2.waitKey(1) 
      # Pass the output (weighted sum through spiking layer, along with previous membrane potential)
      # Output -- spike output + updated potential
      spk_dir_lowerbeta, mem_dir_lowerbeta = snn1_lowerbeta(con_out, mem_dir_lowerbeta)
      img_lyr1 = torch.squeeze(spk_dir_lowerbeta.detach())
      img_lyr1[img_lyr1>0] = 255 #convert to 0 and 1

      # sys.exit(0)
      # mask_input = visual_frame
      # mask_input[(img_lyr1>0)]=0
      mask_input = mask_input_logical(visual_frame, img_lyr1, 0)
      # mask_input = mask_input_logical(mask_input, img_lyr1, 19)
      # mask_input = mask_input_logical(mask_input, img_lyr1, 3)
      sep_frame = np.array(mask_input)
      sep_frame= ((sep_frame - sep_frame.min()) * (1/(sep_frame.max() - sep_frame.min()) * 255)).astype('uint8')
      # cv2.imshow('sep_frame', sep_frame)
      # cv2.waitKey(1)

      spike_frame = np.array(img_lyr1)
      spike_frame= ((spike_frame - spike_frame.min()) * (1/(spike_frame.max() - spike_frame.min()) * 255)).astype('uint8')
      # cv2.imshow('spk_frame', spike_frame)
      # cv2.waitKey(1)

      # monocular frame with Dotie
      dotie_x, dotie_y = np.where(sep_frame > 0)
      dotie = np.zeros((len(dotie_x), 2))
      dotie[:,0], dotie[:,1] = dotie_x, dotie_y
      dotie_image = np.zeros((sep_frame.shape))
      for x, y in dotie:
         dotie_image[int(x),int(y)] = 255

      # cv2.imshow('monocular_dotie_frame', dotie_image)
      # cv2.waitKey(1)

      x0=dotie_x.min()
      y0=dotie_y.min()
      x1=dotie_x.max()
      y1=dotie_y.max()
      cv2.rectangle(dotie_image, (y0, x0), (y1, x1), (255, 255, 0), 1)#, cv2.LINE_AA)

      cx = x0+(x1-x0)//2
      cy = y0+(y1-y0)//2
      cy_m = cy * (4/640) - 2
      cv2.circle(dotie_image, (cy, cx), 5, (255, 255, 0), 1)#, cv2.LINE_AA)

      cv2.putText(dotie_image, "y= %d Pixel" % cy,(dotie_image.shape[1] - 300, dotie_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (255, 255, 0), 3)

      cv2.putText(dotie_image, "x= %d" % cx,(dotie_image.shape[1] - 600, dotie_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (255, 255, 0), 3)

      cv2.putText(dotie_image, "y= %.2fm" % cy_m,(dotie_image.shape[1] - 400, dotie_image.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX,2.0, (255, 255, 0), 3)
      # cv2.imshow('dotie_bounding_box', dotie_image)

def eventLogger(data):#interrupt service routine
   e=data.events
   hardThreshold=False
   global eventFrame
   global eventFrame_viz
   global t
   global t_prev
   global flag
   global img
   global flag
   #imgStream=[args.groupname]['events'][e]
   #print(data)
   kp=1
   kn=1
   interval = 1e6/30 #divide by fps

   for i in e:
      if ((i.ts.secs*1e6)+(i.ts.nsecs/1e3))>(t_prev+interval):
         img = eventFrame
         eventFrame=np.zeros((480, 640), dtype=np.uint8)
         flag = True
         t_prev=((i.ts.secs*1e6)+(i.ts.nsecs/1e3))
      if hardThreshold==False:
         if int(i.polarity)==1:
            eventFrame[int(i.y), int(i.x)]=eventFrame[int(i.y), int(i.x)]+kp
         else: 
            eventFrame[int(i.y), int(i.x)]=eventFrame[int(i.y), int(i.x)]+kp
      else:
         if int(i.polarity)==1:
            eventFrame[int(i.y), int(i.x)]=255#eventFrame[int(i[1]), int(i[0])]+kp
         else:
            #pass
            eventFrame[int(i.y), int(i.x)]=eventFrame[int(i.y), int(i.x)]-kn # (height,width)
            # eventFrame=cv2.cvtColor(eventFrame, cv2.COLOR_GRAY2RGB)
   # sys.exit(0)

   dotie()
   
   
   
   ################################################### PLANNING ############################################################
   
def callback(data):
    global x_target, y_ring_now, y_target, x_drone, depth, vel_x_drone, transforms, model, t_prev, cy_m, shift
    
    desired_yaw_degree = 180
    z_target = 1.5
    x_target = data.pose[2].position.x - 1
    
    y_ring_now = cy_m

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
        Energy.append(energy)    # import pdb; pdb.set_trace()

    
def ev_planner_events():
    rospy.init_node('ev_planner_events', anonymous=True)
 
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
        states = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        # shift = abs(states('bebop2','').pose.position.y)
        shift = 0

        rospy.Subscriber("/bebop2/event_camera/events", EventArray, eventLogger)
        rospy.sleep(1)
        rospy.Subscriber('/bebop2/command/motor_speed', mav_msgs.msg.Actuators, speed_logger, tcp_nodelay=True)
        # input("Model loaded! Press Enter to start")
        ev_planner_events()
    except rospy.ROSInterruptException:
        pass
