cmake_minimum_required(VERSION 3.8)
project(autonomy_planning_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17 -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(lattice_planner_node src/lattice_planner_node.cpp)
ament_target_dependencies(lattice_planner_node rclcpp nav_msgs geometry_msgs std_msgs Eigen3)

add_executable(behavior_planner_node src/behavior_planner_node.cpp)
target_link_libraries(behavior_planner_node nlohmann_json::nlohmann_json)
ament_target_dependencies(behavior_planner_node rclcpp std_msgs nav_msgs)

install(TARGETS
  lattice_planner_node
  behavior_planner_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
