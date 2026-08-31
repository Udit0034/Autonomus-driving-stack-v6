from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='autonomy_evaluation', executable='evaluate_node', output='screen', parameters=[{'use_sim_time': True}]),
    ])
