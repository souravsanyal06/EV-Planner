#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:52:43 2023

@author: sourav
"""

import os

cmd1 = "posy=$(rostopic echo -n 1 /gazebo/model_states/pose[2]/position/y); echo $c | bc | sed -e 's/^-\./-0./' -e 's/^\./0./'"
os.system(cmd1)
cmd2 = "rosrun rotors_gazebo waypoint_publisher -3 $c 1.5 180 __ns:=bebop2"
os.system(cmd2)