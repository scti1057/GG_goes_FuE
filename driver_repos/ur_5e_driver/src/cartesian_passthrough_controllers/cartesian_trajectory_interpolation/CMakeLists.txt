cmake_minimum_required(VERSION 3.5)
project(cartesian_trajectory_interpolation)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(cartesian_passthrough_msgs REQUIRED)

add_library(${PROJECT_NAME}
  src/cartesian_state.cpp
  src/cartesian_trajectory.cpp
  src/cartesian_trajectory_segment.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}
  geometry_msgs
  tf2_eigen
  cartesian_passthrough_msgs
)

# No catkin/ROS1 add_dependencies here.

target_link_libraries(${PROJECT_NAME}
  # nothing extra
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install all public headers (both cartesian_trajectory_interpolation/ and trajectory_interface/)
install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING REGEX ".*\\.h(pp)?"
)

ament_export_include_directories(include)
ament_export_dependencies(geometry_msgs tf2_eigen cartesian_passthrough_msgs geometry_msgs)

ament_package()