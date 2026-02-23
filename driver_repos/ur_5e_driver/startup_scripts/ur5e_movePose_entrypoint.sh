#!/bin/bash

# Source the ROS environment
source /opt/ros/humble/setup.bash

# Rebuild all packages in the workspace to ensure compatibility after mounting
cd /home/ros_ws
colcon build

# Source the workspace again after building
source /home/ros_ws/install/setup.bash

# Start the UR driver bringup for the UR5e robot cell
ros2 launch ur_cell_description ur5e_cell_bringup.launch &
ros2 run irp_ur5e_support ap_movePose.py &

# Keep the container alive
wait
