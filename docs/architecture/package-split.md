# Target package split for the AV6 stack

This repository is currently in a transition phase. The runtime is working as a single ROS workspace, but the long-term structure should be split by responsibility so it stays readable and easier to maintain.

## Target layout

```text
src/
├── ros-carla-msgs/                  # external standard interface package
├── autonomy_perception/            # camera bridge + inference + semantic perception
│   ├── package.xml
│   ├── setup.py
│   ├── launch/
│   └── autonomy_perception/
├── autonomy_localization/          # EKF + VO + SLAM
│   ├── package.xml
│   ├── CMakeLists.txt
│   ├── include/
│   └── src/
├── autonomy_planning/              # mission + VPG + lattice + behavior
│   ├── package.xml
│   ├── CMakeLists.txt
│   ├── include/
│   └── src/
├── autonomy_control/               # controller stack
│   ├── package.xml
│   ├── CMakeLists.txt
│   ├── include/
│   └── src/
├── autonomy_evaluation/            # plots, metrics, validation, analysis
│   ├── package.xml
│   ├── setup.py
│   ├── launch/
│   └── autonomy_evaluation/
└── autonomy_common/                # reusable utilities and shared types
    ├── package.xml
    └── include/
```

## Node-to-package mapping

### Perception
- `carla_node_native.py`
- `infrence_node.py`
- `traffic_light_fusion_node.py`
- `doppler_node.py`
- `semantic_lane_centering.cpp`
- `semantic_occupancy_grid.cpp`
- `lidar_occupancy_grid_node.cpp` (if kept as sensor pre-processing for perception)
- `dashboard_node.py`

### Localization
- `ekf_node.cpp`
- `visual_odometry_node.cpp`
- `ndt_slam_node.cpp`

### Planning
- `mission_planner_node.py`
- `lattice_planner_node.cpp`
- `behavior_planner_node.cpp`
- `vpg_node.py`

### Control
- `carla_controller_node.cpp`

### Evaluation
- `evaluate_node.py`
- plotting and metrics scripts under `eval_plots/` and `analyze_*.py`

## Migration rule

1. Keep the current working packages as the active runtime until the split is complete.
2. Move nodes only when the message interfaces and launch structure are stable.
3. Preserve the ROS topic contracts during the split.
4. Keep `ros-carla-msgs` as an external dependency; do not rewrite it.

## Current status

The repo is already validated in the current workspace and the stack is launching successfully. The split below is the clean target architecture for the next refactor pass.
