cmake_minimum_required(VERSION 3.8)
project(create_ros2bag)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(create_ros2bag src/create_ros2_bag.cpp)

ament_target_dependencies(create_ros2bag
  rclcpp
  sensor_msgs
  rosbag2_cpp
  OpenCV
)

# 実行可能ファイルのインストール
install(TARGETS create_ros2bag
  DESTINATION lib/create_ros2bag
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME})  


ament_package()
