#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import math
import json
from sklearn.cluster import DBSCAN

# ============================================================
# CONFIGURATIONS & EXTRINSICS
# ============================================================
VELOCITY_THRESHOLD = 1.0  
RADAR_MIN_Z = -0.4  
RADAR_MAX_Z = 2.2   

RADAR_CFG = {
    "front": {"x": 2.0, "y": 0.0, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    "rear":  {"x": -2.0, "y": 0.0, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": 180.0},
    "fl":    {"x": 1.8, "y": -0.8, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": -60.0},
    "fr":    {"x": 1.8, "y": 0.8, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": 60.0},
    "rl":    {"x": -1.8, "y": -0.8, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": -125.0},
    "rr":    {"x": -1.8, "y": 0.8, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": 125.0}
}

def get_extrinsics_matrix(cfg):
    roll, pitch, yaw = math.radians(cfg['roll']), math.radians(cfg['pitch']), math.radians(cfg['yaw'])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [ -sp,             cp*sr,             cp*cr]
    ], dtype=np.float32)
    T = np.array([cfg['x'], cfg['y'], cfg['z']], dtype=np.float32).reshape(3, 1)
    return R, T

class DopplerExtractionNode(Node):
    def __init__(self):
        super().__init__('doppler_node')
        self.cb_group = ReentrantCallbackGroup()

        self.v_ego = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.radar_buffers = {key: np.empty((4, 0), dtype=np.float32) for key in RADAR_CFG.keys()}
        self.radar_extrinsics = {key: get_extrinsics_matrix(cfg) for key, cfg in RADAR_CFG.items()}

        self.odom_sub = self.create_subscription(Odometry, '/ekf/odometry', self.odom_callback, 10, callback_group=self.cb_group)
        
        for r_name in RADAR_CFG.keys():
            self.create_subscription(
                PointCloud2, 
                f'/carla/hero/radar_{r_name}', 
                lambda msg, name=r_name: self.radar_callback(msg, name), 
                10, 
                callback_group=self.cb_group
            )

        # 🎯 Raw Data Publisher
        self.target_pub = self.create_publisher(String, '/perception/radar/dynamic_targets', 10)
        self.process_timer = self.create_timer(1.0 / 20.0, self.process_radars)
        
        self.get_logger().info("📡 Doppler Radial Velocity Extraction Node Online.")

    def odom_callback(self, msg: Odometry):
        self.v_ego = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z], dtype=np.float32)

    def radar_callback(self, msg: PointCloud2, name: str):
        try:
            # Note: pc2.read_points is the standard ROS2 Py bottleneck.
            # Keeping skip_nans=True ensures clean data before it hits NumPy.
            data = list(pc2.read_points(msg, skip_nans=True))
            self.radar_buffers[name] = np.array(data, dtype=np.float32).T if data else np.empty((4, 0), dtype=np.float32)
        except Exception:
            self.radar_buffers[name] = np.empty((4, 0), dtype=np.float32)

    def process_radars(self):
        all_dynamic_pts = []
        all_dynamic_vels = []

        for name, pts in self.radar_buffers.items():
            if pts.shape[1] == 0: continue

            p_sensor = pts[0:3, :]  
            measured_v = pts[3, :]  
            R, T = self.radar_extrinsics[name]
            
            # Fast matrix transform
            p_ego = R @ p_sensor + T
            z_ego = p_ego[2, :]
            
            # Z-Filtering
            valid_z = (z_ego > RADAR_MIN_Z) & (z_ego < RADAR_MAX_Z)
            if not np.any(valid_z): continue

            p_sensor = p_sensor[:, valid_z]
            p_ego = p_ego[:, valid_z]
            measured_v = measured_v[valid_z]
            
            # Ego Velocity Compensation (Vectorized)
            dir_vectors = R @ p_sensor 
            norms = np.linalg.norm(dir_vectors, axis=0)
            norms[norms == 0] = 1e-6
            u_vectors = dir_vectors / norms
            
            # Dot product for expected velocity
            expected_v = -(self.v_ego[0]*u_vectors[0] + self.v_ego[1]*u_vectors[1] + self.v_ego[2]*u_vectors[2])
            true_dynamic_v = measured_v - expected_v
            
            # Filter Statics
            dynamic_mask = np.abs(true_dynamic_v) > VELOCITY_THRESHOLD
            
            if np.any(dynamic_mask):
                all_dynamic_pts.append(p_ego[:, dynamic_mask])
                all_dynamic_vels.append(true_dynamic_v[dynamic_mask])

        # Clustering
        targets_payload = []
        if all_dynamic_pts:
            stacked_pts = np.hstack(all_dynamic_pts)
            stacked_vels = np.concatenate(all_dynamic_vels)
            
            xy_pts = stacked_pts[0:2, :].T
            if len(xy_pts) > 0:
                # [FIX APPLIED] min_samples=1 allows single sparse hits to register as targets
                # algorithm='ball_tree' is generally faster for small, low-dimensional spatial datasets
                clustering = DBSCAN(eps=2.0, min_samples=1, algorithm='ball_tree').fit(xy_pts)
                labels = clustering.labels_
                
                # [OPTIMIZATION] np.unique is faster than casting to set()
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    # Noise check (mostly redundant with min_samples=1, kept for safety)
                    if label == -1: continue 
                    
                    mask = (labels == label)
                    # np.median is efficient, and we extract as floats for the JSON payload
                    cluster_x = float(np.median(xy_pts[mask, 0]))
                    cluster_y = float(np.median(xy_pts[mask, 1]))
                    cluster_v = float(np.median(stacked_vels[mask]))
                    
                    targets_payload.append([cluster_x, cluster_y, cluster_v])

        # Publish the lightweight array
        self.target_pub.publish(String(data=json.dumps(targets_payload)))

def main(args=None):
    rclpy.init(args=args)
    node = DopplerExtractionNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()