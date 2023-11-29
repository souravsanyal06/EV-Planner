#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 19:32:47 2023

@author: sourav
"""

import h5py
import numpy as np
from PIL import Image
import cv2
import os

dataset_root_path = "/home/sourav/cv_ws/src/trajectory_learner/scripts/data/"
f = h5py.File(dataset_root_path + "datadummy.hdf5",'r')

depthset = f["dummy"]["depth"][:]

# import pdb; pdb.set_trace()