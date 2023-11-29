#!/usr/bin/env python3
import rospy
# for event camera

# for all sensor data
import sensor_msgs.msg as sms

from rospy.numpy_msg import numpy_msg
import cv2
import numpy as np

from signal import signal, SIGINT
from gazebo_msgs.srv import GetModelState
                                                                                                                        
import h5py

import argparse
from argparse import RawTextHelpFormatter

import snntorch as snn
import torch
import torch.nn as nn

import imutils



def imgCompressor():
   pass

def dataSaver(signal_received, frame):
   # Organise and save data here
   global l
   global img
   global imgts
   global a
   global w
   global acc_cov
   global pos
   global ori
   global evt, rgb, dep, imu, pose

   global eventFrame
   global conv1, snn1_lowerbeta, mem_dir_lowerbeta
   global depts, depth
   
   #evt.unregister()

   depts=np.stack(depts)
   file="/home/sourav/cv_ws/src/trajectory_learner/scripts/data/"+args.hdf5Filename+".hdf5"
   
   f=h5py.File(file, "a")
   g=f.create_group(args.groupname)
   a=np.array(a)
   sz= True
   
   g.create_dataset("depth_ts", data=depts, dtype=depts.dtype, chunks=sz)
   del depts
   print('Depth timestamp shape: '+str(g['depth_ts'].shape))
   print("Depth timestamps saved successfully!")
   depth=np.stack(depth)
   g.create_dataset("depth", data=depth, dtype=depth.dtype, chunks=sz)
   del depth
   print('Depth data shape: '+str(g['depth'].shape))
   print("Depth data saved successfully!")
   
   print("Success!")
   print("Exiting...")


   

def depthLogger(data):
   global depth
   global depts
   global first
   global focalLength
   global KNOWN_WIDTH
   i = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)
   depth.append(i)
   depts.append(data.header.stamp.secs*1e6+data.header.stamp.nsecs/1e3)
   i = cv2.normalize(i, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
   #i = cv2.cvtColor(i, cv2.COLOR_GRAY2RGB)
   # cv2.imshow('depth_frame_raw', i)
   #cv2.waitKey(1)
   #depts.append(data.header.stamp.secs*1e6+data.header.stamp.nsecs/1e3)
   states = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
   x_ring = states('moving_ring','').pose.position.x
   x_drone = states('bebop2','').pose.position.x
   
   if first:
      KNOWN_DISTANCE = abs(x_drone - x_ring) * 39.37  
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
   print("%.2f m" % ((inches / 12)*0.3048))

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
      
      cv2.imshow('depth_frame_box_center', my_image)
      cv2.waitKey(1)
      #print(i.shape)
   
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

def get_iou(ground_truth, pred):
   # coordinates of the area of intersection.
   ix1 = np.maximum(ground_truth[0], pred[0])
   iy1 = np.maximum(ground_truth[1], pred[1])
   ix2 = np.minimum(ground_truth[2], pred[2])
   iy2 = np.minimum(ground_truth[3], pred[3])
     
   # Intersection height and width.
   i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
   i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
     
   area_of_intersection = i_height * i_width
     
   # Ground Truth dimensions.
   gt_height = ground_truth[3] - ground_truth[1] + 1
   gt_width = ground_truth[2] - ground_truth[0] + 1
     
   # Prediction dimensions.
   pd_height = pred[3] - pred[1] + 1
   pd_width = pred[2] - pred[0] + 1
     
   area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
     
   iou = area_of_intersection / area_of_union
     
   return iou



def listener():
   
   # event camera subscriber
   rospy.init_node('listener', anonymous=True)
   # depth camera subscriber
   rospy.init_node('listener', anonymous=True)#change
   rospy.Subscriber("/bebop2/depth_camera/depth/image_raw", numpy_msg(sms.Image), depthLogger)
   # spin() simply keeps python from exiting until this node is stopped
   rospy.spin()

if __name__ == '__main__':
   desc=("Script to generate dataset from a MAVLINK-Gazebo simulation.\nWARNING: The simulation must be running BEFORE this script is executed. Executing this script without the simulation running may lead to unpredictable behaviour.\nIn case the given HDF5 file already contains a group with name 'groupname', this script will fail gracefully, and report the corresponding error. Existing groups can be viewed/deleted using the script 'deleter.py'")
   epilog=("For further documentation, refer the Dataset Generation Suite documentation page at https://github.com/")
   parser=argparse.ArgumentParser(description=desc, epilog=epilog, formatter_class=RawTextHelpFormatter)
   parser.add_argument('hdf5Filename', type=str, default='test', help="Name of HDF5 file containing the dataset")
   parser.add_argument('groupname', type=str, default='test', help="The group to which data from the current simulation is to be saved")
   args=parser.parse_args()
   # s = sched.scheduler(time.time, time.sleep)
   eventFrame=np.zeros((480, 640), dtype=np.uint8)
   signal(SIGINT, dataSaver)
   #print("in main")
   l=[]
   img=[]
   imgts=[]
   depts=[]
   depth=[]
   ptCld=[]
   a=[]
   acc_cov=[]
   w=[]
   pos=[]
   ori=[]
   prev=0
   t_prev=0
   interval=1e6/30
   flag=False
   first=True
   img=np.zeros((480, 640), dtype=np.uint8)#(height,width)
   #call_events()

   # Convolutional layer (3x3) 
   conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=True).to('cpu')
   # Set the weights manually
   conv1.weight = torch.nn.Parameter(torch.tensor(torch.ones_like(conv1.weight)*0.1))#0.15
   with torch.no_grad():
      conv1.weight[0, 0, 1, 1] = 0.15#0.2
   # spiking neuron parameters -- LEAK FACTOR = 3
   snn1_lowerbeta = snn.Leaky(beta=0.1, threshold=1.75, reset_mechanism="subtract")#modify beta 0.3
   mem_dir_lowerbeta = snn1_lowerbeta.init_leaky()

   listener()
  

      
