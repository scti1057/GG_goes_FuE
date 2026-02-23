#!/bin/bash

# Source the ROS2 environment
source /opt/ros/humble/setup.bash

# Source the workspace setup
source /home/ros/ros2_ws/install/setup.bash

# Start the realsense hardware driver
echo "Starting the realsense ros driver..."
ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true \
  enable_depth:=true \
  depth_module.depth_profile:=640x480x15 \
  depth_module.depth_format:=Z16 \
  enable_color:=false \
  depth_module.emitter_enabled:=1 \
  depth_module.enable_auto_exposure:=true \
  spatial_filter.enable:=true \
  temporal_filter.enable:=true \
  hole_filling_filter.enable:=true \
  spatial_filter.filter_magnitude:=2 \
  spatial_filter.filter_smooth_alpha:=0.50 \
  spatial_filter.filter_smooth_delta:=20 \
  temporal_filter.filter_smooth_alpha:=0.40 \
  temporal_filter.filter_smooth_delta:=20 \
  temporal_filter.persistence_control:=2 \
  hole_filling_filter.holes_fill:=2

# Keep container alive
wait