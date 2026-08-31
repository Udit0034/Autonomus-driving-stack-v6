from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='autonomy_control_cpp', executable='carla_controller_node', output='screen', parameters=[{'use_sim_time': True}]),
    ])
