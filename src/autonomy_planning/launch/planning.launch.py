from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Node(package='autonomy_planning', executable='mission_planner_node', output='screen', parameters=[{'use_sim_time': True}]),
        # Node(package='autonomy_planning', executable='vpg_node', output='screen', parameters=[{'use_sim_time': True}]),
        # Node(package='autonomy_planning_cpp', executable='lattice_planner_node', output='screen', parameters=[{'use_sim_time': True}]),
        # Node(package='autonomy_planning_cpp', executable='behavior_planner_node', output='screen', parameters=[{'use_sim_time': True}]),
    ])
