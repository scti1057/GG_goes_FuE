#!/bin/bash

# Ensure that the ROBOT_IP environment variable is set
if [ -z "$ROBOT_IP" ]; then
    echo "Error: ROBOT_IP environment variable is not set."
    exit 1
fi

# Check if an ip address is set for an schunk axia ft sensor
if [ -z "$AXIA_FT_IP" ]; then
    echo "Warning: AXIA_FT_IP environment variable is not set. Proceeding without force-torque sensor."
else
    echo "AXIA_FT_IP is set to $AXIA_FT_IP. Proceeding with force-torque sensor configuration."
    # Set default values if environment variables are not provided
    export AXIA_SENSOR_TYPE="${AXIA_SENSOR_TYPE:-ati_axia}"
    export AXIA_FT_RDT_SAMPLING_RATE="${AXIA_FT_RDT_SAMPLING_RATE:-500}"
    export AXIA_FT_INTERNAL_FILTER_RATE="${AXIA_FT_INTERNAL_FILTER_RATE:-0}"
    export AXIA_USE_HARDWARE_BIASING="${AXIA_USE_HARDWARE_BIASING:-false}"

    echo "Using AXIA_SENSOR_TYPE=$AXIA_SENSOR_TYPE"
    echo "Using AXIA_FT_RDT_SAMPLING_RATE=$AXIA_FT_RDT_SAMPLING_RATE"
    echo "Using AXIA_FT_INTERNAL_FILTER_RATE=$AXIA_FT_INTERNAL_FILTER_RATE"
    echo "Using AXIA_USE_HARDWARE_BIASING=$AXIA_USE_HARDWARE_BIASING"
fi

# Source the ROS environment
source /opt/ros/humble/setup.bash

# Rebuild the necessary packages
cd /home/ros_ws
colcon build --packages-select ur_description ur_robot_driver

# Source the workspace again after building
source /home/ros_ws/install/setup.bash

# Start the UR driver bringup for the UR5e robot cell
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=$ROBOT_IP launch_rviz:=false &

sleep 5

# If AXIA_FT_IP is set, launch the force-torque sensor node
if [ ! -z "$AXIA_FT_IP" ]; then
    echo "Starting force-torque sensor node with IP: $AXIA_FT_IP"
    ros2 launch net_ft_driver net_ft_broadcaster.launch.py \
    ip_address:=$AXIA_FT_IP \
    sensor_type:=$AXIA_SENSOR_TYPE \
    rdt_sampling_rate:=$AXIA_FT_RDT_SAMPLING_RATE \
    internal_filter_rate:=$AXIA_FT_INTERNAL_FILTER_RATE \
    use_hardware_biasing:=$AXIA_USE_HARDWARE_BIASING &
fi

sleep 3

# Start the ap_moveTwist node
ros2 run irp_ur5e_support ap_moveTwist.py &

# Keep the container alive
wait
