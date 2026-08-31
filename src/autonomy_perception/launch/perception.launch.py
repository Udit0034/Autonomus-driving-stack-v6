from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='autonomy_perception', executable='carla_node', output='screen', parameters=[{'use_sim_time': True}]),
        Node(package='autonomy_perception', executable='infrence_node', output='screen', parameters=[{'use_sim_time': True}]),
        Node(package='autonomy_perception', executable='traffic_light_fusion_node', output='screen', parameters=[{'use_sim_time': True}]),
        Node(package='autonomy_perception', executable='doppler_node', output='screen', parameters=[{'use_sim_time': True}]),
        Node(package='autonomy_perception_cpp', executable='lidar_occupancy_grid_node', output='screen', parameters=[{'use_sim_time': True}]),
        Node(package='autonomy_perception_cpp', executable='ttc_fusion_node', output='screen', parameters=[{'use_sim_time': True}]),
        Node(package='autonomy_perception_cpp', executable='semantic_lane_centering', output='screen', parameters=[{'use_sim_time': True}]),
        Node(package='autonomy_perception_cpp', executable='semantic_occupancy_grid', output='screen', parameters=[{'use_sim_time': True}]),
    ])
