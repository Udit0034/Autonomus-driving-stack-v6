cmake_minimum_required(VERSION 3.8)
project(autonomy_localization_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17 -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc video calib3d)
find_package(PCL REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(ekf_node src/ekf_node.cpp)
ament_target_dependencies(ekf_node rclcpp sensor_msgs nav_msgs geometry_msgs message_filters tf2 tf2_ros tf2_geometry_msgs)

add_executable(ndt_slam_node src/ndt_slam_node.cpp)
ament_target_dependencies(ndt_slam_node rclcpp sensor_msgs geometry_msgs cv_bridge nav_msgs pcl_ros pcl_conversions tf2_ros tf2_geometry_msgs)
target_link_libraries(ndt_slam_node ${PCL_LIBRARIES})

add_executable(visual_odometry_node src/visual_odometry_node.cpp)
target_include_directories(visual_odometry_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(visual_odometry_node ${OpenCV_LIBRARIES})
ament_target_dependencies(visual_odometry_node rclcpp sensor_msgs nav_msgs geometry_msgs cv_bridge Eigen3 tf2)

install(TARGETS
  ekf_node
  ndt_slam_node
  visual_odometry_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
