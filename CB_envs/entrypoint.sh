#!/bin/bash

# Start ros master
roscore &

# Waiting for roscore to start
sleep 3

# Working directory
cd ~/realsense-ws/

#run CB node
rosrun yolo ultra.py
