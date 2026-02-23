cmake_minimum_required(VERSION 3.5)
project(irp_ur5e_support)

# Set C++ standard to 14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT WIN32)
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

## Find colcon macros and libraries
## if COMPONENTS list like find_package(colcon REQUIRED COMPONENTS xyz)
## is used, also find other colcon packages
# ROS 2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)           # C++ nodes
find_package(rclpy REQUIRED)            # Python nodes
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(controller_manager REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(irp_ur5e_interfaces REQUIRED)
# find_package(moveit_ros_planning_interface REQUIRED)  # Includes moveit_commander
################################################
## Declare ROS messages, services and actions ##
################################################
#actions are processed similarly to messages, using rosidl_generate_interfaces()
#custom actions are operated or coordinated using rosidl_generate_interfaces()

#################
## Build Setup ##
#################
# Specify additional locations of header files
# Local headers in include/
include_directories(
  include
)
add_executable(pose_publisher src/my_publisher.cpp)

# Link dependencies for the executable
ament_target_dependencies(pose_publisher
  rclcpp
  geometry_msgs
  trajectory_msgs
  std_msgs
)

# Install targets
install(TARGETS
  pose_publisher
   RUNTIME DESTINATION lib/${PROJECT_NAME}
)
#############
## Install ##
#############

# Install Python scripts , copies scripts to runtime location
install(PROGRAMS
  scripts/ap_movePose.py
  scripts/ap_moveTwist.py
  scripts/ap_moveJointsPos.py
  scripts/robot_pose_publisher.py
  scripts/admittanz_regler.py
  scripts/admittanz_regler_publisher.py
  scripts/admittance_force_controller_test.py
  DESTINATION lib/${PROJECT_NAME}
)
# Install C++ executables (if any)
# install(TARGETS pose_publisher
#   RUNTIME DESTINATION lib/${PROJECT_NAME}
# )
install(DIRECTORY
  launch
  config
  urdf
  DESTINATION share/${PROJECT_NAME}/
)

##########################
## Finalize package ######
##########################
ament_package()
# This is the ROS 2 equivalent of catkin_package().