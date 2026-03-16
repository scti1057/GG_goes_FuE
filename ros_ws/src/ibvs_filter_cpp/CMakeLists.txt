cmake_minimum_required(VERSION 3.8)
project(ibvs_filter_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ibvs_msgs REQUIRED)

add_library(ibvs_filter_core
  src/base_filter.cpp
  src/filters.cpp
)

target_include_directories(ibvs_filter_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ibvs_filter_core
  Eigen3::Eigen
  ${OpenCV_LIBS}
)

ament_target_dependencies(ibvs_filter_core
  rclcpp
)

add_executable(filter_node src/filter_node.cpp)
target_link_libraries(filter_node ibvs_filter_core)
ament_target_dependencies(filter_node
  rclcpp
  rcl_interfaces
  geometry_msgs
  sensor_msgs
  std_msgs
  ibvs_msgs
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS ibvs_filter_core filter_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()
