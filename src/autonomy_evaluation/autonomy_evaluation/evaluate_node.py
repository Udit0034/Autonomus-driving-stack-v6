import os
import json
import time
import math
import collections
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy import signal
from sklearn.metrics import confusion_matrix
from matplotlib.lines import Line2D

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Header, Float32
from sensor_msgs.msg import Image, Imu, NavSatFix
from nav_msgs.msg import Odometry, Path as NavPath
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import message_filters

# ==========================================
# CONFIGURATION
# ==========================================
NUM_CLASSES = 30
CAM_CAPS = {
    'front_left': 250.0,
    'rear': 150.0,
    'side_left': 80.0,
    'side_right': 80.0,
    'front_right': 250.0
}
CAM_RESOLUTIONS = {
    'front_left': (720, 1280),
    'front_right': (720, 1280),
    'rear': (400, 800),
    'side_left': (400, 800),
    'side_right': (400, 800)
}

# ==========================================
# MATH UTILITIES
# ==========================================
def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def angle_error(est, gt):
    return wrap_angle(est - gt)

def quaternion_to_euler(w, x, y, z):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return wrap_angle(yaw), wrap_angle(pitch), wrap_angle(roll)

def print_traj_info(name, df, x_col, y_col, z_col=None):
    try:
        if df is None or df.empty:
            print(f"{name}: No data")
            return

        if x_col not in df.columns or y_col not in df.columns:
            print(f"{name}: Missing columns")
            return

        valid = df[[x_col, y_col]].dropna()

        if valid.empty:
            print(f"{name}: No valid points")
            return

        start_x = valid.iloc[0][x_col]
        start_y = valid.iloc[0][y_col]

        msg = (
            f"{name}: "
            f"points={len(valid)}, "
            f"start=({start_x:.3f}, {start_y:.3f}"
        )

        if z_col is not None and z_col in df.columns:
            z_valid = df[z_col].dropna()
            if not z_valid.empty:
                msg += f", {z_valid.iloc[0]:.3f}"

        msg += ")"
        print(msg)

    except Exception as e:
        print(f"{name}: Error reading trajectory ({e})") 

# ==========================================
# UNIFIED EVALUATION NODE (NATIVE ROS 2)
# ==========================================
class EvaluateNode(Node):
    def __init__(self):
        super().__init__('evaluate_node')
        
        self.declare_parameter('debug', False)
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        
        self.declare_parameter('tune', False)
        self.tune = self.get_parameter('tune').get_parameter_value().bool_value
        
        self.bridge = CvBridge()
        self.callback_group = ReentrantCallbackGroup()
        
        self.stats = {cam: {'rmse': [], 'd1': [], 'miou': []} for cam in CAM_CAPS.keys()}
        self.frame_counts = {cam: 0 for cam in CAM_CAPS.keys()}
        self.last_timestamps = {cam: None for cam in CAM_CAPS.keys()}
        self.fps_records = {cam: collections.deque(maxlen=30) for cam in CAM_CAPS.keys()}
        
        # ODOMETRY & PLANNING BUFFERS
        self.gt_data = []
        self.ekf_data = []
        self.vo_data = []
        self.slam_data = []
        self.global_path_data = []
        self.local_path_data = []
        self.vgp_data = []
        
        # 🎯 THE MASTER CLOCK
        self.current_gt_time = 0.0
        
        # RAW SENSOR BUFFERS
        self.raw_imu_data = []
        self.raw_alt_data = []
        self.raw_wheel_data = []
        self.raw_gnss_data = []
        self.gnss_origin = None
        
        # Decoupled Cache
        self.eval_cache = {cam: collections.OrderedDict() for cam in CAM_CAPS.keys()}
        self.latest_gt = {cam: {'gt_d': None, 'gt_s': None} for cam in CAM_CAPS.keys()}

        for cam in CAM_CAPS.keys():
            self.create_subscription(Image, f'/inference/{cam}/depth', lambda msg, c=cam: self.cache_callback(msg, c, 'd', False), qos_profile_sensor_data, callback_group=self.callback_group)
            self.create_subscription(Image, f'/inference/{cam}/seg', lambda msg, c=cam: self.cache_callback(msg, c, 's', False), qos_profile_sensor_data, callback_group=self.callback_group)

            if self.debug:
                self.create_subscription(Image, f'/carla/hero/depth_{cam}/image', lambda msg, c=cam: self.cache_callback(msg, c, 'd', True), qos_profile_sensor_data, callback_group=self.callback_group)
                self.create_subscription(Image, f'/carla/hero/seg_{cam}/image', lambda msg, c=cam: self.cache_callback(msg, c, 's', True), qos_profile_sensor_data, callback_group=self.callback_group)
        
        # ODOMETRY SUBSCRIPTIONS
        self.create_subscription(Odometry, '/carla/hero/odometry', self.gt_odom_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        self.create_subscription(Odometry, '/ekf/odometry', self.ekf_odom_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        self.create_subscription(Odometry, '/vo/odometry', self.vo_odom_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        self.create_subscription(Odometry, '/slam/odometry', self.slam_odom_callback, qos_profile_sensor_data, callback_group=self.callback_group)

        # PLANNING & CONTROL SUBSCRIPTIONS
        self.create_subscription(NavPath, '/planning/global_path', self.global_path_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        self.create_subscription(NavPath, '/planning/local_path', self.local_path_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        self.create_subscription(Float32, '/planning/target_speed', self.vgp_callback, 10, callback_group=self.callback_group)

        # RAW SENSOR SUBSCRIPTIONS (Now ALWAYS ON to ensure Data is logged)
        self.create_subscription(Imu, '/carla/hero/imu', self.tune_imu_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        self.create_subscription(PointStamped, '/carla/altimeter', self.tune_alt_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        self.create_subscription(Odometry, '/carla/hero/wheel_odom', self.tune_wheel_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        
        self.gnss_front_sub = message_filters.Subscriber(self, NavSatFix, '/carla/hero/gnss_front')
        self.gnss_rear_sub = message_filters.Subscriber(self, NavSatFix, '/carla/hero/gnss_rear')
        self.gnss_sync = message_filters.ApproximateTimeSynchronizer([self.gnss_front_sub, self.gnss_rear_sub], queue_size=10, slop=0.05)
        self.gnss_sync.registerCallback(self.tune_gnss_callback)

        self.get_logger().info(f"🧪 Unified Evaluation Node Started . Debug mode: {self.debug}")
        self.get_logger().info("✅ Raw Sensor Data (GNSS, IMU, Wheel) enabled for Plot 1 Trajectory Tracing.")
        self.get_logger().info("Plots will be generated upon Ctrl+C (destroy_node).")

    # ==========================================
    # PATH PLANNING & VELOCITY CALLBACKS
    # ==========================================
    def vgp_callback(self, msg: Float32):
        if self.current_gt_time == 0.0: return
        self.vgp_data.append({'Timestamp': self.current_gt_time, 'Target_Speed': msg.data})

    def global_path_callback(self, msg: NavPath):
        pts = [{'X': p.pose.position.x, 'Y': p.pose.position.y} for p in msg.poses]
        
        if len(pts) > 0:
            # Only update and log if the path actually changed
            if pts != self.global_path_data:
                self.global_path_data = pts
                self.get_logger().info(
                    f"🗺️ DEBUG: Evaluator intercepted NEW/UPDATED Global Path with {len(pts)} points."
                )
        else:
            # Keep the throttle here so it doesn't spam if it gets stuck empty
            self.get_logger().warn(
                "⚠️ DEBUG: Evaluator intercepted an EMPTY Global Path!", 
                throttle_duration_sec=60.0
            )

    def local_path_callback(self, msg: NavPath):
        if self.current_gt_time == 0.0: return
        t = self.current_gt_time
        path_pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if len(path_pts) == 0:
            self.get_logger().warn("⚠️ DEBUG: Evaluator intercepted an EMPTY Local Lattice Path!")
        self.local_path_data.append({'Timestamp': t, 'path': path_pts})

    # ==========================================
    # TUNING DATA COLLECTION CALLBACKS
    # ==========================================
    def latlon_to_xy(self, lat, lon):
        EARTH_RADIUS = 6378137.0
        if self.gnss_origin is None: return 0.0, 0.0
        lat0, lon0 = self.gnss_origin[0], self.gnss_origin[1]
        
        dlat = math.radians(lat - lat0)
        dlon = math.radians(lon - lon0)
        
        x = dlon * math.cos(math.radians(lat0)) * EARTH_RADIUS
        y = dlat * EARTH_RADIUS
        return x, y

    def tune_gnss_callback(self, front_msg: NavSatFix, rear_msg: NavSatFix):
        t = front_msg.header.stamp.sec + front_msg.header.stamp.nanosec * 1e-9
        # Save exact Lat/Lon/Alt. No math. No conversions.
        self.raw_gnss_data.append({
            'Timestamp': t,
            'Front_Lat': front_msg.latitude, 'Front_Lon': front_msg.longitude, 'Front_Alt': front_msg.altitude,
            'Rear_Lat': rear_msg.latitude, 'Rear_Lon': rear_msg.longitude, 'Rear_Alt': rear_msg.altitude
        })

    def tune_imu_callback(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # Save exact Quaternions, Accel, and Gyro. No negations. No Euler math.
        self.raw_imu_data.append({
            'Timestamp': t, 
            'Accel_X': msg.linear_acceleration.x, 'Accel_Y': msg.linear_acceleration.y, 'Accel_Z': msg.linear_acceleration.z,
            'Gyro_X': msg.angular_velocity.x, 'Gyro_Y': msg.angular_velocity.y, 'Gyro_Z': msg.angular_velocity.z,
            'Qw': msg.orientation.w, 'Qx': msg.orientation.x, 'Qy': msg.orientation.y, 'Qz': msg.orientation.z
        })

    def tune_alt_callback(self, msg: PointStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.raw_alt_data.append({'Timestamp': t, 'Alt_Z': msg.point.z})

    def tune_wheel_callback(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.raw_wheel_data.append({'Timestamp': t, 'Odom_Velocity': msg.twist.twist.linear.x})

    # ==========================================
    # ASYNC STATE-BASED CACHE LOGIC
    # ==========================================
    def cache_callback(self, msg, cam, modality, is_gt):
        if is_gt:
            self.latest_gt[cam][f'gt_{modality}'] = msg
            return

        t_key = f"{msg.header.stamp.sec}_{msg.header.stamp.nanosec}"

        if t_key not in self.eval_cache[cam]:
            self.eval_cache[cam][t_key] = {'pred_d': None, 'pred_s': None}

        self.eval_cache[cam][t_key][f'pred_{modality}'] = msg
        frame_data = self.eval_cache[cam][t_key]

        if frame_data['pred_d'] is not None and frame_data['pred_s'] is not None:
            if self.debug:
                if self.latest_gt[cam]['gt_d'] is not None and self.latest_gt[cam]['gt_s'] is not None:
                    frame_data['gt_d'] = self.latest_gt[cam]['gt_d']
                    frame_data['gt_s'] = self.latest_gt[cam]['gt_s']
                    self.process_evaluation(cam, frame_data)
                    if t_key in self.eval_cache[cam]:
                        del self.eval_cache[cam][t_key]
            else:
                self.process_evaluation(cam, frame_data)
                if t_key in self.eval_cache[cam]:
                    del self.eval_cache[cam][t_key]

        if len(self.eval_cache[cam]) > 5:
            self.eval_cache[cam].popitem(last=False)

    def decode_depth(self, msg):
        if isinstance(msg, np.ndarray):
            bgra = msg.astype(np.float64)
            b, g, r = bgra[:, :, 0], bgra[:, :, 1], bgra[:, :, 2]
            normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
            return (normalized * 1000.0).astype(np.float32)

        if msg.encoding in ['bgra8', 'rgba8']:
            bgra = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            b, g, r = bgra[:, :, 0].astype(np.float64), bgra[:, :, 1].astype(np.float64), bgra[:, :, 2].astype(np.float64)
            normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
            return (normalized * 1000.0).astype(np.float32)

        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def decode_seg(self, msg):
        if isinstance(msg, np.ndarray):
            return msg[:, :, 2]
        if msg.encoding in ['bgra8', 'rgba8']:
            bgra = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            return bgra[:, :, 2] 
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="8UC1")

    def compute_metrics(self, pred_d, gt_d, pred_s, gt_s, max_depth):
        mask = (gt_d >= 1.0) & (gt_d <= max_depth) & (pred_d >= 1.0) & (pred_d <= max_depth)
        rmse, d1 = np.nan, np.nan
        
        if np.any(mask):
            p, g = pred_d[mask], gt_d[mask]
            rmse = np.sqrt(((p - g)**2).mean())
            ratio = np.maximum(p/g, g/p)
            d1 = (ratio < 1.25).mean() * 100.0
        
        miou = np.nan
        if gt_s is not None and pred_s is not None:
            valid_s = (gt_s != 255) & (gt_s < NUM_CLASSES)
            if np.any(valid_s):
                gt_valid = gt_s[valid_s].astype(np.int32)
                pred_valid = pred_s[valid_s].astype(np.int32)
                
                cm = np.bincount(NUM_CLASSES * gt_valid + pred_valid, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
                
                intersection = np.diag(cm)
                union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
                
                ious = []
                for i in range(1, NUM_CLASSES):  
                    if union[i] > 0:
                        ious.append(intersection[i] / union[i])
                
                if ious: miou = np.mean(ious)

        del pred_d, gt_d, pred_s, gt_s
        return float(rmse), float(d1), float(miou)
    
    def process_evaluation(self, cam_name, frame_data):
        try:
            now = time.perf_counter()
            if self.last_timestamps[cam_name] is not None:
                dt = now - self.last_timestamps[cam_name]
                if dt > 0: self.fps_records[cam_name].append(1.0 / dt)
            self.last_timestamps[cam_name] = now
            self.frame_counts[cam_name] += 1

            if not self.debug:
                return

            pred_d = self.bridge.imgmsg_to_cv2(frame_data['pred_d'], desired_encoding="32FC1")
            pred_s = self.bridge.imgmsg_to_cv2(frame_data['pred_s'], desired_encoding="8UC1")
            gt_d = self.decode_depth(frame_data['gt_d'])
            gt_s = self.decode_seg(frame_data['gt_s'])

            if pred_d.shape != gt_d.shape:
                pred_d = cv2.resize(pred_d, (gt_d.shape[1], gt_d.shape[0]), interpolation=cv2.INTER_LINEAR)
            if pred_s.shape != gt_s.shape:
                pred_s = cv2.resize(pred_s, (gt_s.shape[1], gt_s.shape[0]), interpolation=cv2.INTER_NEAREST)

            max_cap = CAM_CAPS[cam_name]
            rmse, d1, miou = self.compute_metrics(pred_d, gt_d, pred_s, gt_s, max_depth=max_cap)

            if not np.isnan(rmse): self.stats[cam_name]['rmse'].append(rmse)
            if not np.isnan(d1):   self.stats[cam_name]['d1'].append(d1)
            if not np.isnan(miou): self.stats[cam_name]['miou'].append(miou)

        except Exception as e:
            self.get_logger().error(f"Evaluation failed for {cam_name}: {e}")

    # ==========================================
    # ODOMETRY DATA GATHERING (ALWAYS ON)
    # ==========================================
    def gt_odom_callback(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.current_gt_time = t  # 🎯 SET THE MASTER CLOCK

        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        yaw, pitch, roll = quaternion_to_euler(q.w, q.x, q.y, q.z)
        speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)

        self.gt_data.append({
            'Timestamp': t, 'Loc_X': pos.x, 'Loc_Y': pos.y, 'Loc_Z': pos.z,
            'Yaw_Degrees': math.degrees(yaw), 
            'Pitch_Degrees': math.degrees(pitch), 
            'Roll_Degrees': math.degrees(roll),   
            'GT_Velocity': speed
        })

    def ekf_odom_callback(self, msg: Odometry):
        if self.current_gt_time == 0.0: return
        t = self.current_gt_time  # 🎯 OVERWRITE WITH GT TIME

        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        yaw, pitch, roll = quaternion_to_euler(q.w, q.x, q.y, q.z)
        
        cov_x = msg.pose.covariance[0] if len(msg.pose.covariance) == 36 else 0.0
        cov_y = msg.pose.covariance[7] if len(msg.pose.covariance) == 36 else 0.0

        bias_x = msg.twist.twist.angular.x
        bias_y = msg.twist.twist.angular.y
        bias_z = msg.twist.twist.angular.z

        self.ekf_data.append({
            'Timestamp': t, 'Est_X': pos.x, 'Est_Y': pos.y, 'Est_Z': pos.z,
            'Est_Yaw': yaw, 'Est_Pitch': pitch, 'Est_Roll': roll,
            'Est_Qw': q.w, 'Est_Qx': q.x, 'Est_Qy': q.y, 'Est_Qz': q.z,
            'Est_Vx': v.x, 'Est_Vy': v.y, 'P_dp_x': cov_x, 'P_dp_y': cov_y,
            'Est_Bias_Gx': bias_x, 'Est_Bias_Gy': bias_y, 'Est_Bias_Gz': bias_z
        })

    def slam_odom_callback(self, msg: Odometry):
        if self.current_gt_time == 0.0: return
        t = self.current_gt_time  # 🎯 OVERWRITE WITH GT TIME

        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw, _, _ = quaternion_to_euler(q.w, q.x, q.y, q.z)
        
        self.slam_data.append({
            'Timestamp': t, 
            'SLAM_X': pos.x, 
            'SLAM_Y': pos.y, 
            'SLAM_Z': pos.z,
            'SLAM_Yaw': yaw
        })

    def vo_odom_callback(self, msg: Odometry):
        if self.current_gt_time == 0.0: return
        t = self.current_gt_time  # 🎯 OVERWRITE WITH GT TIME

        pos = msg.pose.pose.position
        self.vo_data.append({
            'Timestamp': t, 
            'VO_X': pos.x, 
            'VO_Y': pos.y, 
            'VO_Z': pos.z
        })

    # ==========================================
    # SHUTDOWN ROUTINE (METRICS & PLOTTING)
    # ==========================================
    def destroy_node(self):
        self.get_logger().info(f"📊 DEBUG COUNTS -> Global Path Pts: {len(self.global_path_data)} | Local Path Msgs: {len(self.local_path_data)} | VGP Msgs: {len(self.vgp_data)}")
        self.get_logger().warn(f"DEBUG COUNTS -> GT: {len(self.gt_data)} | EKF: {len(self.ekf_data)} | SLAM: {len(self.slam_data)} | VO: {len(self.vo_data)}")
        
        # 1. EVALUATE PERCEPTION
        if self.debug:
            self.get_logger().info("🛑 RUN FINISHED: CALCULATING FINAL AVERAGE PERCEPTION METRICS")
            self.get_logger().info("="*60)
            final_metrics = {}
            print(f"\n{'CAMERA':<15} | {'mIoU (0-1)':<12} | {'RMSE (m)':<12} | {'δ < 1.25 (%)':<12} | {'Frames'}")
            print("-" * 75)

            for cam in CAM_CAPS.keys():
                cam_stats = self.stats[cam]
                avg_miou = np.mean(cam_stats['miou']) if len(cam_stats['miou']) > 0 else 0.0
                avg_rmse = np.mean(cam_stats['rmse']) if len(cam_stats['rmse']) > 0 else 0.0
                avg_d1   = np.mean(cam_stats['d1']) if len(cam_stats['d1']) > 0 else 0.0
                
                final_metrics[cam] = {
                    "mIoU": avg_miou, "RMSE": avg_rmse, "delta_1_25": avg_d1,
                    "max_depth_cap": CAM_CAPS[cam], "frames_evaluated": self.frame_counts[cam]
                }
                print(f"{cam:<15} | {avg_miou:<12.4f} | {avg_rmse:<12.2f} | {avg_d1:<12.2f} | {self.frame_counts[cam]}")

            print("=" * 75)
            save_path = os.path.join(os.getcwd(), 'perception_metrics.json')
            try:
                with open(save_path, 'w') as f: json.dump(final_metrics, f, indent=4)
                self.get_logger().info(f"💾 Perception metrics successfully saved to: {save_path}")
            except Exception as e:
                self.get_logger().error(f"❌ Failed to save perception metrics: {e}")
                
        else:
            self.get_logger().info("🛑 Inference Node Shutdown. Final Camera Framerates:")
            print(f"\n{'CAMERA':<15} | {'Avg FPS':<10} | {'Frames':<10}")
            print("-" * 40)
            for cam in CAM_CAPS.keys():
                avg_fps = float(np.mean(self.fps_records[cam])) if len(self.fps_records[cam]) > 0 else 0.0
                print(f"{cam:<15} | {avg_fps:<10.2f} | {self.frame_counts[cam]:<10}")
            print("=" * 40)

        # 2. EVALUATE ODOMETRY & GENERATE PLOTS
        if len(self.gt_data) > 0 and len(self.ekf_data) > 0:
            self.generate_ekf_plots()
        else:
            self.get_logger().warn("⚠️ Not enough Odometry data collected to generate plots.")

        # 3. EXPORT TUNING DATA
        if self.tune and len(self.raw_imu_data) > 0:
            tune_dir = os.path.join(os.getcwd(), 'eval_plots', 'tuning_data')
            os.makedirs(tune_dir, exist_ok=True)
            
            pd.DataFrame(self.raw_imu_data).to_csv(os.path.join(tune_dir, 'imu_data.csv'), index=False)
            pd.DataFrame(self.raw_alt_data).to_csv(os.path.join(tune_dir, 'altimeter_data.csv'), index=False)
            pd.DataFrame(self.raw_wheel_data).to_csv(os.path.join(tune_dir, 'odom_data.csv'), index=False)
            
            if len(self.raw_gnss_data) > 0:
                pd.DataFrame(self.raw_gnss_data).to_csv(os.path.join(tune_dir, 'gnss_data.csv'), index=False)
            
            if len(self.slam_data) > 0:
                pd.DataFrame(self.slam_data).to_csv(os.path.join(tune_dir, 'slam_data.csv'), index=False)
            if len(self.vo_data) > 0:
                pd.DataFrame(self.vo_data).to_csv(os.path.join(tune_dir, 'vo_data.csv'), index=False)
            if len(self.gt_data) > 0:
                pd.DataFrame(self.gt_data).to_csv(os.path.join(tune_dir, 'gt_data.csv'), index=False)
            if len(self.ekf_data) > 0:
                pd.DataFrame(self.ekf_data).to_csv(os.path.join(tune_dir, 'ekf_data.csv'), index=False)
            self.get_logger().info(f"✅ Full Tuning Dataset exported to {tune_dir}")

        super().destroy_node()

    def generate_ekf_plots(self):
        if len(self.gt_data) < 5 or len(self.ekf_data) < 5:
            self.get_logger().warn("⚠️ Not enough EKF/GT data collected to generate plots.")
            return
        

        run_dir = os.path.join(os.getcwd(), 'eval_plots')
        os.makedirs(run_dir, exist_ok=True)

        self.get_logger().info(f"\n--- Generating Sensor Fusion & Planning Suite in {run_dir} ---")

        sns.set_theme(style="darkgrid", palette="deep")

        # 🎯 Average out any duplicate messages that arrived during the exact same GT tick
        # 🎯 Average out any duplicate messages that arrived during the exact same GT tick
        odom_df = pd.DataFrame(self.gt_data).drop_duplicates(subset=['Timestamp']).reset_index(drop=True)
        # BUGFIX: Changed .mean() to .last() to prevent destroying angles near +/- 180 (Pi)
        m1_df = pd.DataFrame(self.ekf_data).groupby('Timestamp').last().reset_index()
        
        slam_df = pd.DataFrame(self.slam_data).groupby('Timestamp').last().reset_index() if self.slam_data else pd.DataFrame(columns=['Timestamp', 'SLAM_X', 'SLAM_Y', 'SLAM_Z'])
        vo_df = pd.DataFrame(self.vo_data).groupby('Timestamp').last().reset_index() if self.vo_data else pd.DataFrame(columns=['Timestamp', 'VO_X', 'VO_Y', 'VO_Z'])
        # 🎯 EXACT MERGE
        try:
            merged = pd.merge(m1_df, odom_df, on='Timestamp', how='inner')
            if not slam_df.empty:
                merged = pd.merge(merged, slam_df, on='Timestamp', how='inner')
            if not vo_df.empty:
                merged = pd.merge(merged, vo_df, on='Timestamp', how='inner')
                
            # Merge VGP Target Speed into the main dataframe
            # Merge VGP Target Speed into the main dataframe
            vgp_df = pd.DataFrame(self.vgp_data).groupby('Timestamp').last().reset_index() if self.vgp_data else pd.DataFrame(columns=['Timestamp', 'Target_Speed'])          
            if not vgp_df.empty:
                merged = pd.merge(merged, vgp_df, on='Timestamp', how='left').ffill()
                merged['Target_Speed'] = merged['Target_Speed'].fillna(method='bfill').fillna(0)
                
        except Exception as e:
            self.get_logger().error(f"Merge failed: {e}")
            return
        
        if merged.empty:
            self.get_logger().error("Merge failed: Dataframes emptied. Ensure sensors are publishing.")
            return

        # 🎯 FILTER THE GRACE PERIOD (Using pure Sim Time)
        settling_time = 5.0
        merged = merged[merged['Timestamp'] >= settling_time].reset_index(drop=True)

        if merged.empty:
            self.get_logger().warn("⚠️ Run was too short to generate metric logic.")
            return

        # 🎯 NORMALIZE PLOT TIME (Starts exactly at T=0.0s)
        start_time = merged['Timestamp'].min()
        merged['Timestamp'] -= start_time

        # # ==========================================
        # # 1. INDEPENDENTLY ALIGN ALL TRAJECTORIES
        # # ==========================================
        # # Align EKF
        # offset_ekf_x = merged['Loc_X'].iloc[0] - merged['Est_X'].iloc[0]
        # offset_ekf_y = merged['Loc_Y'].iloc[0] - merged['Est_Y'].iloc[0]
        # offset_ekf_z = merged['Loc_Z'].iloc[0] - merged['Est_Z'].iloc[0]

        # merged['Est_X'] += offset_ekf_x
        # merged['Est_Y'] += offset_ekf_y
        # merged['Est_Z'] += offset_ekf_z

        # # Also align m1_df for standalone usage in other plots if needed
        # m1_df['Est_X'] += offset_ekf_x
        # m1_df['Est_Y'] += offset_ekf_y
        # m1_df['Est_Z'] += offset_ekf_z
        
        # # Align SLAM
        # if 'SLAM_X' in merged.columns and not merged['SLAM_X'].isna().all():
        #     offset_slam_x = merged['Loc_X'].iloc[0] - merged['SLAM_X'].iloc[0]
        #     offset_slam_y = merged['Loc_Y'].iloc[0] - merged['SLAM_Y'].iloc[0]
        #     offset_slam_z = merged['Loc_Z'].iloc[0] - merged['SLAM_Z'].iloc[0]
            
        #     merged['SLAM_X'] += offset_slam_x
        #     merged['SLAM_Y'] += offset_slam_y
        #     merged['SLAM_Z'] += offset_slam_z
            
        # # Align VO
        # if 'VO_X' in merged.columns and not merged['VO_X'].isna().all():
        #     offset_vo_x = merged['Loc_X'].iloc[0] - merged['VO_X'].iloc[0]
        #     offset_vo_y = merged['Loc_Y'].iloc[0] - merged['VO_Y'].iloc[0]
        #     offset_vo_z = merged['Loc_Z'].iloc[0] - merged['VO_Z'].iloc[0]
            
        #     merged['VO_X'] += offset_vo_x
        #     merged['VO_Y'] += offset_vo_y
        #     merged['VO_Z'] += offset_vo_z

        # ==========================================
        # CALCULATE 3D ERRORS FOR ALL SENSORS
        # ==========================================
        # EKF Error
        merged['error_x'] = merged['Est_X'] - merged['Loc_X']
        merged['error_y'] = merged['Est_Y'] - merged['Loc_Y']
        merged['error_z'] = merged['Est_Z'] - merged['Loc_Z']
        merged['pos_error'] = np.sqrt(merged['error_x']**2 + merged['error_y']**2 + merged['error_z']**2)
        
        # SLAM Error
        if 'SLAM_X' in merged.columns and not merged['SLAM_X'].isna().all():
            merged['slam_pos_error'] = np.sqrt((merged['SLAM_X'] - merged['Loc_X'])**2 + 
                                               (merged['SLAM_Y'] - merged['Loc_Y'])**2 + 
                                               (merged['SLAM_Z'] - merged['Loc_Z'])**2)
        # VO Error
        if 'VO_X' in merged.columns and not merged['VO_X'].isna().all():
            merged['vo_pos_error'] = np.sqrt((merged['VO_X'] - merged['Loc_X'])**2 + 
                                             (merged['VO_Y'] - merged['Loc_Y'])**2 + 
                                             (merged['VO_Z'] - merged['Loc_Z'])**2)
        
        merged['yaw_error'] = merged.apply(lambda row: angle_error(row['Est_Yaw'], math.radians(row['Yaw_Degrees'])), axis=1)
        merged['pitch_error'] = merged.apply(lambda row: angle_error(-row['Est_Pitch'], math.radians(row['Pitch_Degrees'])), axis=1)
        merged['roll_error'] = merged.apply(lambda row: angle_error(-row['Est_Roll'], math.radians(row['Roll_Degrees'])), axis=1)  # unchanged
        merged['quat_norm'] = np.sqrt(merged['Est_Qw']**2 + merged['Est_Qx']**2 + merged['Est_Qy']**2 + merged['Est_Qz']**2)

        dt = merged['Timestamp'].diff().where(lambda x: x > 0.01, 0.01) 
        velocity_series = merged['GT_Velocity'].values
        
        if len(velocity_series) > 21:  
            smoothed_vel = merged['GT_Velocity'].rolling(window=11, center=True, min_periods=1).mean()
            try:
                smoothed_vel = pd.Series(signal.savgol_filter(smoothed_vel, window_length=21, polyorder=3), index=merged.index)
            except:
                pass
        else:
            smoothed_vel = merged['GT_Velocity'].rolling(window=20, center=True, min_periods=1).mean()
        
        smoothed_vel = smoothed_vel.ffill().bfill()
        merged['Long_Accel'] = smoothed_vel.diff() / dt
        merged['Long_Jerk'] = merged['Long_Accel'].diff() / dt
        
        yaw_rate_rad = np.radians(merged['Yaw_Degrees'].diff() / dt)
        merged['Lat_Accel'] = smoothed_vel * yaw_rate_rad
        merged['Lat_Jerk'] = merged['Lat_Accel'].diff() / dt

        valid_data = merged[merged['Timestamp'] > 2.0]
        abs_jerk = valid_data['Long_Jerk'].abs().dropna()

        # ==========================================
        # PLOT 1: Trajectory Fusion & 3D Error 
        # ==========================================
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
        
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(odom_df['Loc_X'], odom_df['Loc_Y'], color='black', linestyle='--', label='Ground Truth', linewidth=2)
        
        if 'VO_X' in merged.columns and not merged['VO_X'].isna().all():
            ax1.plot(merged['VO_X'], merged['VO_Y'], color='#2ca02c', linestyle='-', label='Raw Visual Odometry', linewidth=2, alpha=0.7)
            
        if 'SLAM_X' in merged.columns and not merged['SLAM_X'].isna().all():
            ax1.plot(merged['SLAM_X'], merged['SLAM_Y'], color='#d62728', linestyle=':', label='Raw LiDAR SLAM', linewidth=2, alpha=0.9)
            
        
        # Raw GNSS
        if len(self.raw_gnss_data) > 0:
            gnss_df = pd.DataFrame(self.raw_gnss_data)
            # Convert raw Lat/Lon to XY on the fly just for plotting
            lat0, lon0 = gnss_df['Front_Lat'].iloc[0], gnss_df['Front_Lon'].iloc[0]
            EARTH_RADIUS = 6378137.0
            
            gnss_x = np.radians(gnss_df['Front_Lon'] - lon0) * math.cos(math.radians(lat0)) * EARTH_RADIUS
            gnss_y = np.radians(gnss_df['Front_Lat'] - lat0) * EARTH_RADIUS
            
            offset_gnss_x = merged['Loc_X'].iloc[0] - gnss_x.iloc[0]
            offset_gnss_y = merged['Loc_Y'].iloc[0] - gnss_y.iloc[0]
            ax1.plot(gnss_x + offset_gnss_x, gnss_y + offset_gnss_y, color='#ff7f0e', linestyle='-.', label='Raw GNSS', alpha=0.6)

        # Basic Dead-Reckoning (Wheel Odometry Velocity + IMU Yaw)
        if len(self.raw_wheel_data) > 0 and len(self.raw_imu_data) > 0:
            w_df = pd.DataFrame(self.raw_wheel_data).set_index('Timestamp')
            i_df = pd.DataFrame(self.raw_imu_data).set_index('Timestamp')
            
            # Calculate Compass Yaw from raw Quaternions on the fly
            def get_compass(row):
                w, x, y, z = row['Qw'], row['Qx'], row['Qy'], row['Qz']
                yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                compass = -(yaw + math.pi / 4.0)   # 45°, not 90°
                return (compass + math.pi) % (2.0 * math.pi) - math.pi
                
            i_df['Compass'] = i_df.apply(get_compass, axis=1)
            
            dr_df = pd.concat([w_df, i_df]).sort_index().ffill().dropna()
            
            if not dr_df.empty:
                dr_df['dt'] = dr_df.index.to_series().diff().fillna(0)
                dr_df['dt'] = np.clip(dr_df['dt'], 0, 0.1) # Cap extreme dt
                
                dr_df['dx'] = dr_df['Odom_Velocity'] * np.cos(dr_df['Compass']) * dr_df['dt']
                dr_df['dy'] = dr_df['Odom_Velocity'] * np.sin(dr_df['Compass']) * dr_df['dt']
                
                dr_df['DR_X'] = dr_df['dx'].cumsum() + merged['Loc_X'].iloc[0]
                dr_df['DR_Y'] = dr_df['dy'].cumsum() + merged['Loc_Y'].iloc[0]
                ax1.plot(dr_df['DR_X'], dr_df['DR_Y'], color='brown', linestyle=':', label='Raw IMU + Wheel Odom (DR)', alpha=0.8)

        ax1.plot(merged['Est_X'], merged['Est_Y'], color='#1f77b4', linestyle='-', label='EKF Fused Trajectory', linewidth=2)
        
        start_x, start_y = merged['Loc_X'].iloc[0], merged['Loc_Y'].iloc[0]
        end_x, end_y = merged['Loc_X'].iloc[-1], merged['Loc_Y'].iloc[-1]
        
        ax1.scatter(start_x, start_y, marker='o', color='green', s=120, zorder=5, label='Start', edgecolor='white')
        ax1.scatter(end_x, end_y, marker='X', color='red', s=120, zorder=5, label='End', edgecolor='white')

        ax1.set_aspect('equal', adjustable='box')
        
        # Hard cap limits exactly at ±50m based on Ground Truth bounds
        # gt_min_x, gt_max_x = merged['Loc_X'].min(), merged['Loc_X'].max()
        # gt_min_y, gt_max_y = merged['Loc_Y'].min(), merged['Loc_Y'].max()
        # ax1.set_xlim([gt_min_x - 50.0, gt_max_x + 50.0])
        # ax1.set_ylim([gt_min_y - 50.0, gt_max_y + 50.0])

        ax1.set_title('Trajectory Sensor Fusion Comparison')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.legend(loc='best')

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(merged['Timestamp'], merged['pos_error'], color='#1f77b4', label='EKF Error', linewidth=2)
        
        if 'VO_X' in merged.columns and not merged['VO_X'].isna().all():
            ax2.plot(merged['Timestamp'], merged['vo_pos_error'], color='#2ca02c', label='VO Error', linewidth=2, alpha=0.6)
            
        if 'SLAM_X' in merged.columns and not merged['SLAM_X'].isna().all():
            ax2.plot(merged['Timestamp'], merged['slam_pos_error'], color='#d62728', label='SLAM Error', linewidth=2, alpha=0.6)
            
        ax2.set_title('3D Localization Error Over Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error (m)')
        ax2.set_ylim([0.0, 25.0]) # Hard capped Y-axis
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, '1_trajectory_and_error.png'), dpi=120)
        plt.close()
        
        # ==========================================
        # DEBUG: Print trajectory stats
        # =========================================
        print("\n========== TRAJECTORY SUMMARY ==========")

        print_traj_info("Ground Truth", odom_df, "Loc_X", "Loc_Y", "Loc_Z")
        print_traj_info("EKF", m1_df, "Est_X", "Est_Y", "Est_Z")

        if 'VO_X' in merged.columns:
            print_traj_info("VO", merged, "VO_X", "VO_Y", "VO_Z")

        if self.slam_data:
            slam_print_df = pd.DataFrame(self.slam_data)
            print_traj_info("SLAM", slam_print_df, "SLAM_X", "SLAM_Y", "SLAM_Z")
        else:
            print("SLAM: No data (node crashed)")
        print("========================================\n")

        # ==========================================
        # PLOT 2: Altitude Tracking
        # ==========================================
        plt.figure(figsize=(12, 4))
        plt.plot(merged['Timestamp'], merged['Loc_Z'], color='black', linestyle='--', label='Ground Truth Z', linewidth=2)
        plt.plot(merged['Timestamp'], merged['Est_Z'], color='#1f77b4', linestyle='-', label='EKF Z', linewidth=2, alpha=0.8)
        plt.fill_between(merged['Timestamp'], merged['Loc_Z'], merged['Est_Z'], alpha=0.2, color='gray')
        plt.title('Altitude Tracking (Z-Axis)')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, '2_altitude_tracking.png'), dpi=120)
        plt.close()

        # ==========================================
        # PLOT 3: Orientation Analysis
        # ==========================================
        fig, axes = plt.subplots(3, 2, figsize=(16, 10))
        est_yaw, gt_yaw = np.degrees(merged['Est_Yaw']), merged['Yaw_Degrees']
        est_pitch, gt_pitch = np.degrees(-merged['Est_Pitch']), merged['Pitch_Degrees']
        est_roll, gt_roll = np.degrees(-merged['Est_Roll']), merged['Roll_Degrees']   # unchanged

        axes[0, 0].plot(merged['Timestamp'], gt_yaw, 'k--', label='GT')
        axes[0, 0].plot(merged['Timestamp'], est_yaw, color='#1f77b4', label='EKF')
        axes[0, 0].set_title('Tracking')
        axes[0, 0].set_ylabel('Yaw (°)')
        axes[0, 0].legend()

        axes[1, 0].plot(merged['Timestamp'], gt_pitch, 'k--')
        axes[1, 0].plot(merged['Timestamp'], est_pitch, color='#2ca02c')
        axes[1, 0].set_ylabel('Pitch (°)')

        axes[2, 0].plot(merged['Timestamp'], gt_roll, 'k--')
        axes[2, 0].plot(merged['Timestamp'], est_roll, color='#d62728')
        axes[2, 0].set_ylabel('Roll (°)')

        axes[0, 1].plot(merged['Timestamp'], np.degrees(merged['yaw_error']), color='#1f77b4')
        axes[0, 1].set_title('Error')
        axes[0, 1].set_ylabel('Error (°)')
        
        axes[1, 1].plot(merged['Timestamp'], np.degrees(merged['pitch_error']), color='#2ca02c')
        axes[1, 1].set_ylabel('Error (°)')
        
        axes[2, 1].plot(merged['Timestamp'], np.degrees(merged['roll_error']), color='#d62728')
        axes[2, 1].set_ylabel('Error (°)')

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, '3_orientation_analysis.png'), dpi=120)
        plt.close()

        # ==========================================
        # PLOT 4: Gyro Bias Convergence
        # ==========================================
        has_3axis = all(col in merged.columns for col in ['Est_Bias_Gx', 'Est_Bias_Gy', 'Est_Bias_Gz'])

        if has_3axis:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            for bias_col, ax, label, color in [
                (merged['Est_Bias_Gx'], axes[0], 'Gyro Bias X', '#9467bd'),
                (merged['Est_Bias_Gy'], axes[1], 'Gyro Bias Y', '#e377c2'),
                (merged['Est_Bias_Gz'], axes[2], 'Gyro Bias Z', '#17becf')]:
                ax.plot(merged['Timestamp'], bias_col, color=color, linewidth=2, alpha=0.8)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.set_title(label)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Bias (rad/s)')
        else:
            plt.figure(figsize=(10, 4))
            plt.plot(merged['Timestamp'], merged.get('Est_Gyro_Bias', np.zeros(len(merged))), color='#9467bd', linewidth=2, alpha=0.8)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.title('Gyro Bias Estimation Convergence')
            plt.xlabel('Time (s)')
            plt.ylabel('Bias (rad/s)')

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, '4_gyro_bias_convergence.png'), dpi=120)
        plt.close()

        # ==========================================
        # PLOT 5: Jerk Analysis
        # ==========================================
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
        
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(merged['Timestamp'], merged['Long_Jerk'], color='#ff7f0e', alpha=0.9, linewidth=2)
        ax1.axhline(y=3, color='green', linestyle='--', alpha=0.6, label='Comfort Limit')
        ax1.axhline(y=-3, color='green', linestyle='--', alpha=0.6)
        ax1.set_title('Longitudinal Jerk Over Time')
        ax1.set_ylim(-15, 15)
        ax1.legend()

        ax2 = fig.add_subplot(gs[1])
        h = ax2.hist2d(merged['Lat_Jerk'].fillna(0).clip(-10, 10), merged['Long_Jerk'].fillna(0).clip(-10, 10), bins=40, cmap='mako', range=[[-10, 10], [-10, 10]])
        fig.colorbar(h[3], ax=ax2, label='Frequency')
        ax2.set_title('2D Jerk Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, '5_jerk_analysis.png'), dpi=120)
        plt.close()

        # ==========================================
        # PLOT 6: Spatial Error Map & Covariance
        # ==========================================
        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(1, 2)
        
        ax1 = fig.add_subplot(gs[0])
        scatter = ax1.scatter(merged['Est_X'], merged['Est_Y'], c=merged['pos_error'], cmap='flare_r', s=30, zorder=3)
        ax1.plot(odom_df['Loc_X'], odom_df['Loc_Y'], 'k--', alpha=0.4, label='Ground Truth', zorder=2)
        fig.colorbar(scatter, ax=ax1, label='3D Position Error (m)')
        ax1.set_title('EKF 2D Error Map')
        ax1.set_aspect('equal', adjustable='datalim')

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(merged['Est_X'], merged['Est_Y'], color=(237/255, 59/255, 178/255), label='Estimated Path', linewidth=2, zorder=2)
        for idx, row in merged.iloc[::30].iterrows():
            if row['P_dp_x'] > 0 and row['P_dp_y'] > 0:
                width, height = 2 * 2 * np.sqrt([row['P_dp_x'], row['P_dp_y']])
                ellipse = Ellipse((row['Est_X'], row['Est_Y']), width, height, angle=0, alpha=0.3, color="#098931", zorder=1)
                ax2.add_patch(ellipse)
        ax2.set_title('Position Covariance Ellipses (2σ)')
        ax2.set_aspect('equal', adjustable='datalim')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, '6_spatial_error_and_covariance.png'), dpi=120)
        plt.close()

        # ==========================================
        # PLOT 7: Speed Tracking (Including VGP)
        # ==========================================
        plt.figure(figsize=(12, 5))
        ekf_speed = np.sqrt(merged['Est_Vx']**2 + merged['Est_Vy']**2)
        plt.plot(merged['Timestamp'], ekf_speed, color='#1f77b4', linewidth=2, label='EKF Speed', alpha=0.9)
        plt.plot(merged['Timestamp'], merged['GT_Velocity'], color='black', linestyle='--', linewidth=2, label='GT Speed', alpha=0.7)
        
        if 'Target_Speed' in merged.columns:
            plt.plot(merged['Timestamp'], merged['Target_Speed'], color='purple', linestyle=':', linewidth=2, label='VGP Target Speed (Planner Constraint)', alpha=0.9)

        plt.fill_between(merged['Timestamp'], ekf_speed, merged['GT_Velocity'], alpha=0.2, color='gray')
        plt.title('Velocity Profile Tracking vs EKF & Ground Truth')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, '7_speed_tracking.png'), dpi=120)
        plt.close()

        # ==========================================
        # PLOT 8: PATH PLANNING COMPARISON
        # ==========================================
        fig8 = plt.figure(figsize=(14, 10))
        ax8 = fig8.add_subplot(111)

        ax8.plot(merged['Loc_X'], merged['Loc_Y'], color='black', linestyle='--', label='Ground Truth Executed', linewidth=2, zorder=3)
        ax8.plot(merged['Est_X'], merged['Est_Y'], color='#1f77b4', linestyle='-', label='EKF Executed', linewidth=2, zorder=4)

        if self.global_path_data:
            gp_df = pd.DataFrame(self.global_path_data)
            ax8.plot(gp_df['X'], gp_df['Y'], color='purple', linestyle='-', linewidth=4, label='Mission Global Path', alpha=0.5, zorder=2)

        if self.local_path_data and not merged.empty:
            lp_df = pd.DataFrame(self.local_path_data)
            
            for idx, row in lp_df.iloc[::25].iterrows():
                t_norm = row['Timestamp'] - start_time
                if t_norm < 0: continue
                
                # BUGFIX: Use the 'merged' dataframe which has the time-normalized and offset-aligned EKF state.
                ekf_idx = (merged['Timestamp'] - t_norm).abs().argmin()
                ekf_state = merged.iloc[ekf_idx]
                ex, ey, eyaw = ekf_state['Est_X'], ekf_state['Est_Y'], ekf_state['Est_Yaw']
                
                gx_list, gy_list = [], []
                for lx, ly in row['path']:
                    gx = ex + lx * math.cos(eyaw) + ly * math.sin(eyaw)
                    gy = ey + lx * math.sin(eyaw) - ly * math.cos(eyaw)
                    gx_list.append(gx)
                    gy_list.append(gy)
                    
                ax8.plot(gx_list, gy_list, color='orange', alpha=0.6, linewidth=1.5, zorder=5)
            
            # Custom legend for local path proxy
            custom_line = Line2D([0], [0], color='orange', alpha=0.6, lw=1.5)
            handles, labels = ax8.get_legend_handles_labels()
            handles.append(custom_line)
            labels.append('Local Lattice Plan Traces')
            ax8.legend(handles=handles, labels=labels, loc='best')
        else:
            ax8.legend(loc='best')

        ax8.set_title('Path Planning & Execution Comparison')
        ax8.set_xlabel('X (m)')
        ax8.set_ylabel('Y (m)')
        ax8.set_aspect('equal', adjustable='box')
        
        # ax8.set_xlim([gt_min_x - 50.0, gt_max_x + 50.0])
        # ax8.set_ylim([gt_min_y - 50.0, gt_max_y + 50.0])

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, '8_path_planning_comparison.png'), dpi=120)
        plt.close()

        # ==========================================
        # GENERATE CORE METRICS JSON
        # ==========================================
        duration = merged['Timestamp'].max() - merged['Timestamp'].min()
        
        metrics = {
            "duration_s": float(duration),
            "samples": len(merged),
            "rmse_pos_3d": float(np.sqrt(np.mean(merged['pos_error']**2))),
            "rmse_x": float(np.sqrt(np.mean(merged['error_x']**2))),
            "rmse_y": float(np.sqrt(np.mean(merged['error_y']**2))),
            "rmse_z": float(np.sqrt(np.mean(merged['error_z']**2))),
            "max_error_3d": float(merged['pos_error'].max()),
            "mean_error_3d": float(merged['pos_error'].mean()),
            "rmse_yaw_deg": float(np.sqrt(np.mean(np.degrees(merged['yaw_error'])**2))),
            "rmse_pitch_deg": float(np.sqrt(np.mean(np.degrees(merged['pitch_error'])**2))),
            "rmse_roll_deg": float(np.sqrt(np.mean(np.degrees(merged['roll_error'])**2))),
            "avg_speed": float(merged['GT_Velocity'].mean()),
            "max_jerk": float(abs_jerk.max()) if not abs_jerk.empty else 0.0,
        }
        
        if 'SLAM_X' in merged.columns and not merged['SLAM_X'].isna().all():
            slam_err_x = merged['SLAM_X'] - merged['Loc_X']
            slam_err_y = merged['SLAM_Y'] - merged['Loc_Y']
            slam_err_z = merged['SLAM_Z'] - merged['Loc_Z']
            slam_pos_error = np.sqrt(slam_err_x**2 + slam_err_y**2 + slam_err_z**2)
            metrics['slam_rmse_3d'] = float(np.sqrt(np.mean(slam_pos_error**2)))

        if has_3axis:
            metrics["final_bias_gx"] = float(merged['Est_Bias_Gx'].iloc[-1])
            metrics["final_bias_gy"] = float(merged['Est_Bias_Gy'].iloc[-1])
            metrics["final_bias_gz"] = float(merged['Est_Bias_Gz'].iloc[-1])
            metrics["final_bias_magnitude"] = float(np.sqrt(metrics["final_bias_gx"]**2 + metrics["final_bias_gy"]**2 + metrics["final_bias_gz"]**2))

        metrics_path = os.path.join(run_dir, 'ekf_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        self.get_logger().info(f"✅ Generated 8 Dashboard-Style plots in {run_dir}/")
        self.get_logger().info(f"✅ Evaluated {int(metrics['samples'])} samples over {metrics['duration_s']:.1f}s")
        self.get_logger().info(f"✅ Final 3D RMSE: {metrics['rmse_pos_3d']:.3f}m | Yaw RMSE: {metrics['rmse_yaw_deg']:.3f}°")

def main(args=None):
    rclpy.init(args=args)
    node = EvaluateNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # 1. Do all the heavy plotting FIRST
        node.destroy_node()
        
        # 2. Shut down ROS gracefully
        if rclpy.ok():
            rclpy.shutdown()
            
        # 3. 🎯 THE FIX: Force an immediate, hard exit. 
        # This prevents the MultiThreadedExecutor from hanging on stuck image threads
        # and kills the node before ROS2 gets impatient and throws the SIGKILL error.
        os._exit(0)

if __name__ == '__main__':
    main()