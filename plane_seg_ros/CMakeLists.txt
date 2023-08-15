cmake_minimum_required(VERSION 3.0.2)

project(plane_seg_ros)

set(CMAKE_C_STANDARD 11)  # OpenCV4 is using _Atomic, a C11 extension
set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  eigen_conversions
  pcl_conversions
  tf_conversions
  tf2_eigen
  tf2_ros
  pcl_ros
  grid_map_core
  grid_map_ros
  grid_map_msgs
  plane_seg
  rospy
  message_generation
)

find_package(OpenCV 3.0 QUIET)

add_message_files(
  FILES
  IntList.msg
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS eigen_conversions pcl_conversions tf_conversions pcl_ros plane_seg CATKIN_DEPENDS rospy message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(plane_seg_ros_n1 src/plane_seg_ros_n1.cpp)
target_link_libraries(plane_seg_ros_n1 PUBLIC ${catkin_LIBRARIES})

catkin_install_python(PROGRAMS src/plane_seg_rospy.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(plane_seg_ros_n3 src/plane_seg_ros_n3.cpp)
target_link_libraries(plane_seg_ros_n3 PUBLIC ${catkin_LIBRARIES})

# TODO: Install
