#!/bin/bash

# Source the ROS environment
source /opt/ros/humble/setup.bash

# Rebuild all packages in the workspace to ensure compatibility after mounting
cd /home/ros_ws
colcon build

# Source the workspace again after building
source /home/ros_ws/install/setup.bash

# Start the UR driver bringup for the UR5e robot cell
ros2 launch irp_ur5e_support main.launch.py &

ros2 run irp_ur5e_support ap_moveTwist &
ros2 run irp_ur5e_support ap_movePose &

wait
