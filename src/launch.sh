#!/usr/bin/env bash
source /opt/ros/noetic/setup.bash
source ~/EV-Planner/devel/setup.bash
source ~/killross.sh
roslaunch rotors_gazebo test.launch lockstep:=true &
#rosrun keyboard keyboard &
# rosrun rviz rviz &

