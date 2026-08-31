from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='autonomy_localization_cpp', executable='ekf_node', output='screen', parameters=[{'use_sim_time': True}]),
        Node(package='autonomy_localization_cpp', executable='visual_odometry_node', output='screen', parameters=[{'use_sim_time': True}]),
        Node(package='autonomy_localization_cpp', executable='ndt_slam_node', output='screen', parameters=[{'use_sim_time': True}]),
    ])
