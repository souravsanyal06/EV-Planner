#!/usr/bin/env python3
import rospy
# for event camera
from dvs_msgs.msg import EventArray
# for all sensor data
import sensor_msgs.msg as sms
from rospy.numpy_msg import numpy_msg
import cv2
import numpy as np
from signal import signal, SIGINT
import sys                                                                                                                            
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

def eventLogger(data):#interrupt service routine
   e=data.events
   hardThreshold=False
   global eventFrame
   global eventFrame_viz
   global t
   global t_prev
   global flag
   global img
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
   if first:
      KNOWN_DISTANCE = 78.74
      KNOWN_WIDTH =10.0#check again
      marker = find_marker(i)
      focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
      first=False


   marker = find_marker(i)
   box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
   box = np.int0(box)
   #print([box])
   cv2.drawContours(i, [box], -1, (255, 255, 0), 2)

   inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
   print("%.2f m" % ((inches / 12)*0.3048))
   #cv2.putText(my_image, "%.2fm" % ((inches / 12)*0.3048),(i.shape[1] - 200, i.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX,2.0, (255, 255, 0), 3)
   #sys.exit(0)
   #cv2.putText(i, "%.2fm" % ((inches / 12)*0.3048),(i.shape[1] - 200, i.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (255, 255, 0), 3)
   # cv2.imshow('depth_frame_box', i)
   #cv2.waitKey(1)

   find_center=False
   if find_center:
      Ximage_x, Ximage_y = np.where(i > 0)
      X = np.zeros((len(Ximage_x), 2))
      X[:,0], X[:,1] = Ximage_x, Ximage_y
      my_image = np.zeros((i.shape))
      for x, y in X:
         my_image[int(x),int(y)] = 255
      
      # cv2.imshow('depth_frame_white', my_image)
      # cv2.waitKey(1)
      x0=Ximage_x.min() 
      y0=Ximage_y.min()
      x1=Ximage_x.max()
      y1=Ximage_y.max()
      #cv2.rectangle(my_image, (y0, x0), (y1, x1), (255, 255, 0), 1)
      cx = x0+(x1-x0)//2
      cy = y0+(y1-y0)//2
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

def dotie():
   #print(l)
   global img
   global flag
   global conv1, snn1_lowerbeta, mem_dir_lowerbeta
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

      # cv2.imshow('monocular_event_frame', my_image)
      # cv2.waitKey(1)

      # db = DBSCAN(eps=6, min_samples=10)
      # db.fit(X)
      # y_pred = db.fit_predict(X)
      # # loop around all cluster labels
      # no_labels = y_pred.max()+1

      # for lbl in range(no_labels):
      #    # print(lbl)
      #    y_lbl_3 = np.where(y_pred == lbl)
      #    x_vals = Ximage_x[y_lbl_3]
      #    y_vals = Ximage_y[y_lbl_3]
      #    X_hull = np.column_stack((x_vals, y_vals))
      #    hull_for_this = ConvexHull(X_hull)   
      #    c = np.array(X_hull[hull_for_this.vertices], dtype = np.int32)
      #    # swap indices
      #    c[:,[1,0]]=c[:,[0,1]]
      #    cv2.drawContours(my_image, [c], 0, (255,255,0), 2)
      #    cv2.rectangle(my_image, (y_vals.min(), x_vals.min()), (y_vals.max(), x_vals.max()), (255, 255, 0), 1, cv2.LINE_AA)
      
      #comment below and use top code for db scan and filtering
      if not my_image.any():
         return
      x0=Ximage_x.min()
      y0=Ximage_y.min()
      x1=Ximage_x.max()
      y1=Ximage_y.max()
      cv2.rectangle(my_image, (y0, x0), (y1, x1), (255, 255, 0), 1)#, cv2.LINE_AA)
      vanish=False
      show_always=True

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
         cy_m = cy * (2/640)
         cv2.circle(my_image, (cy, cx), 5, (255, 255, 0), 1)#, cv2.LINE_AA)

         cv2.putText(my_image, "y= %d Pixel" % cy,(my_image.shape[1] - 300, my_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (255, 255, 0), 3)

         cv2.putText(my_image, "x= %d" % cx,(my_image.shape[1] - 600, my_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (255, 255, 0), 3)

         cv2.putText(my_image, "y= %.2fm" % cy_m,(my_image.shape[1] - 400, my_image.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX,2.0, (255, 255, 0), 3)
         cv2.imshow('without_dotie_bounding_box', my_image)
         cv2.waitKey(1)

      # uncomment below for without dotie and make show_always variable true 
      sys.exit(0)

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
      cy_m = cy * (2/640)
      cv2.circle(dotie_image, (cy, cx), 5, (255, 255, 0), 1)#, cv2.LINE_AA)

      cv2.putText(dotie_image, "y= %d Pixel" % cy,(dotie_image.shape[1] - 300, dotie_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (255, 255, 0), 3)

      cv2.putText(dotie_image, "x= %d" % cx,(dotie_image.shape[1] - 600, dotie_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,2.0, (255, 255, 0), 3)

      cv2.putText(dotie_image, "y= %.2fm" % cy_m,(dotie_image.shape[1] - 400, dotie_image.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX,2.0, (255, 255, 0), 3)
      cv2.imshow('dotie_bounding_box', dotie_image)
      # cv2.waitKey(1)
      

def listener():
   
   # event camera subscriber
   rospy.init_node('listener', anonymous=True)
   rospy.Subscriber("/bebop2/event_camera/events", EventArray, eventLogger)
   print("event data listened")
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
  

      
