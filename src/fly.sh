#!/bin/bash



rosrun rotors_gazebo waypoint_publisher  0 0 0 180 0 __ns:=bebop2
rosrun rotors_gazebo waypoint_publisher  0 0 1 180 0 __ns:=bebop2
rosrun rotors_gazebo waypoint_publisher  1 0 1 180 0 __ns:=bebop2
rosrun rotors_gazebo waypoint_publisher  1 1 1 180 0 __ns:=bebop2
rosrun rotors_gazebo waypoint_publisher  2 1 1 180 0 __ns:=bebop2
rosrun rotors_gazebo waypoint_publisher  2 2 1 180 0 __ns:=bebop2
rosrun rotors_gazebo waypoint_publisher  2 2 2 180 0 __ns:=bebop2
rosrun rotors_gazebo waypoint_publisher  1 2 2 180 0 __ns:=bebop2
rosrun rotors_gazebo waypoint_publisher  1 1 2 180 0 __ns:=bebop2
rosrun rotors_gazebo waypoint_publisher  1 1 1 180 0 __ns:=bebop2
rosrun rotors_gazebo waypoint_publisher  1 0 1 180 0 __ns:=bebop2
rosrun rotors_gazebo waypoint_publisher  0 0 1 180 0 __ns:=bebop2
rosrun rotors_gazebo waypoint_publisher  0 0 0 180 0 __ns:=bebop2

