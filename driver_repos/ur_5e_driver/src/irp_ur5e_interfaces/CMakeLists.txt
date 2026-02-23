cmake_minimum_required(VERSION 3.5)
project(irp_ur5e_interfaces)

# Find the standard message packages first (they must be found before rosidl_generate_interfaces)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(action_msgs REQUIRED)

# Find the ROSIDL generators
find_package(rosidl_default_generators REQUIRED)

# Generate the action interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ApmovePose.action"
  "action/ApmoveTwist.action"
  "action/ApmoveJointsPos.action"
  "action/AdmittanzRegler.action"

  DEPENDENCIES 
  geometry_msgs 
  std_msgs 
  trajectory_msgs
  action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
