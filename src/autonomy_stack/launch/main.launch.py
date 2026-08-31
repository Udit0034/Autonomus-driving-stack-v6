import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    debug_arg = DeclareLaunchArgument('debug', default_value='false')
    rviz_arg = DeclareLaunchArgument('with_rviz', default_value='true')
    tune_arg = DeclareLaunchArgument('tune', default_value='false')

    debug_config = LaunchConfiguration('debug')
    with_rviz = LaunchConfiguration('with_rviz')
    tune_config = LaunchConfiguration('tune')
    display_available = 'true' if os.environ.get('DISPLAY') else 'false'

    carla_node = Node(
        package='autonomy_perception',
        executable='carla_node',
        output='screen',
        parameters=[{'debug': debug_config, 'use_sim_time': True}]
    )

    ekf_node = Node(package='autonomy_localization_cpp', executable='ekf_node', output='screen', parameters=[{'use_sim_time': True}])
    vo_node = Node(package='autonomy_localization_cpp', executable='visual_odometry_node', output='screen', parameters=[{'use_sim_time': True}])
    slam_node = Node(
        package='autonomy_localization_cpp',
        executable='ndt_slam_node',
        name='slam_node',
        output='screen',
        parameters=[{
            'voxel_size': 0.75,
            'ndt_resolution': 3.0,
            'ndt_max_iterations': 30,
            'ndt_step_size': 0.1,
            'ndt_transformation_epsilon': 0.01,
            'keyframe_distance': 1.0,
            'map_file': '/home/ubuntu/AV6/repo-dev/maps/hero_map.pcd',
            'mapping_mode': True,
            'map_extend_radius': 5.0,
            'use_sim_time': True
        }]
    )

    inference_node = Node(package='autonomy_perception', executable='infrence_node', output='screen', parameters=[{'debug_mode': debug_config, 'use_sim_time': True}])
    traffic_light_fusion_node = Node(package='autonomy_perception', executable='traffic_light_fusion_node', output='screen', parameters=[{'use_sim_time': True}])
    doppler_node = Node(package='autonomy_perception', executable='doppler_node', output='screen', parameters=[{'use_sim_time': True}])
    ttc_fusion_node = Node(package='autonomy_perception_cpp', executable='ttc_fusion_node', output='screen', parameters=[{'use_sim_time': True}])
    lidar_occupancy_grid_node = Node(package='autonomy_perception_cpp', executable='lidar_occupancy_grid_node', output='screen', parameters=[{'use_sim_time': True}])
    semantic_occupancy_grid_node = Node(package='autonomy_perception_cpp', executable='semantic_occupancy_grid', output='screen', parameters=[{'use_sim_time': True}])
    lane_centering_node = Node(package='autonomy_perception_cpp', executable='semantic_lane_centering', output='screen', parameters=[{'use_sim_time': True}])

    mission_planner_node = Node(package='autonomy_planning', executable='mission_planner_node', output='screen', parameters=[{'use_sim_time': True}])
    vpg_node = Node(package='autonomy_planning', executable='vpg_node', output='screen', parameters=[{'use_sim_time': True}])
    lattice_planner_node = Node(package='autonomy_planning_cpp', executable='lattice_planner_node', output='screen', parameters=[{'use_sim_time': True}])
    behavior_planner_node = Node(package='autonomy_planning_cpp', executable='behavior_planner_node', output='screen', parameters=[{'use_sim_time': True}])

    carla_controller_node = Node(
        package='autonomy_control_cpp',
        executable='carla_controller_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    evaluate_node = Node(
        package='autonomy_evaluation',
        executable='evaluate_node',
        output='screen',
        parameters=[{'debug': debug_config, 'tune': tune_config, 'use_sim_time': True}]
    )

    dashboard_node = Node(
        package='autonomy_perception',
        executable='dashboard_node',
        output='screen',
        parameters=[{'debug': debug_config, 'use_sim_time': True}],
        condition=IfCondition(with_rviz)
    )

    eval_debug_rviz = os.path.join(get_package_share_directory('autonomy_stack'), 'rviz', 'eval_debug.rviz')
    eval_inference_rviz = os.path.join(get_package_share_directory('autonomy_stack'), 'rviz', 'eval_inference.rviz')
    rviz_config_file = PythonExpression(["'", debug_config, "' == 'true' and '", eval_debug_rviz, "' or '", eval_inference_rviz, "'"])
    rviz_condition = PythonExpression(["'", with_rviz, "' == 'true' and '", display_available, "' == 'true'"])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(rviz_condition)
    )

    return LaunchDescription([
        debug_arg, rviz_arg, tune_arg,
        carla_node,
        ekf_node,
        vo_node,
        slam_node,
        inference_node, traffic_light_fusion_node, doppler_node, ttc_fusion_node, lidar_occupancy_grid_node,
        semantic_occupancy_grid_node,
        lane_centering_node,
        mission_planner_node, lattice_planner_node, vpg_node, behavior_planner_node,
        carla_controller_node,
        evaluate_node, dashboard_node, rviz_node
    ])
