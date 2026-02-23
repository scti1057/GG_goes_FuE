cmake_minimum_required(VERSION 3.5)
project(cartesian_passthrough_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

set(msg_files
  msg/CartesianPosture.msg
  msg/CartesianTolerance.msg
  msg/CartesianTrajectory.msg
  msg/CartesianTrajectoryPoint.msg
)

set(action_files
  action/FollowCartesianTrajectory.action
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  ${action_files}
  DEPENDENCIES
    action_msgs
    builtin_interfaces
    geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
