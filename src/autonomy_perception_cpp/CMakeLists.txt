cmake_minimum_required(VERSION 3.8)
project(autonomy_perception_cpp)

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
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc video calib3d)
find_package(nlohmann_json REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(lidar_occupancy_grid_node src/lidar_occupancy_grid_node.cpp)
ament_target_dependencies(lidar_occupancy_grid_node rclcpp sensor_msgs nav_msgs Eigen3)

add_executable(ttc_fusion_node src/ttc_fusion_node.cpp)
target_include_directories(ttc_fusion_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ttc_fusion_node nlohmann_json::nlohmann_json ${OpenCV_LIBRARIES})
ament_target_dependencies(ttc_fusion_node rclcpp std_msgs Eigen3 nav_msgs sensor_msgs cv_bridge)

add_executable(semantic_lane_centering src/semantic_lane_centering.cpp)
target_include_directories(semantic_lane_centering PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(semantic_lane_centering ${OpenCV_LIBRARIES})
ament_target_dependencies(semantic_lane_centering rclcpp sensor_msgs std_msgs cv_bridge message_filters Eigen3)

add_executable(semantic_occupancy_grid src/semantic_occupancy_grid.cpp)
target_include_directories(semantic_occupancy_grid PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(semantic_occupancy_grid ${OpenCV_LIBRARIES})
ament_target_dependencies(semantic_occupancy_grid rclcpp sensor_msgs nav_msgs cv_bridge Eigen3)

install(TARGETS
  lidar_occupancy_grid_node
  ttc_fusion_node
  semantic_lane_centering
  semantic_occupancy_grid
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
