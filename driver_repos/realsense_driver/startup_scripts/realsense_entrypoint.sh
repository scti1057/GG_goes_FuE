#!/bin/bash

# Source the ROS2 environment
source /opt/ros/humble/setup.bash

# Source the workspace setup
source /home/ros/ros2_ws/install/setup.bash

# Start the realsense hardware driver
echo "Starting the realsense ros driver..."
ros2 launch realsense2_camera rs_launch.py align_depth:=true

# Keep the container alive
wait
