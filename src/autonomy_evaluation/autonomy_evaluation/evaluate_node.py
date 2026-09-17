import os
import sys
import csv
import json
import time
import math
import shutil
import collections
import subprocess
import numpy as np
import cv2
import pandas as pd

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32
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

# ==========================================
# MATH UTILITIES
# ==========================================
def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

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

# ==========================================
# UNIFIED EVALUATION NODE
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
        
        # 1. CLEANUP OLD RUN DATA (Prevents teleporting plot lines)
        self.base_dir = os.path.join(os.getcwd(), 'eval_temp')
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
            
        # 2. CREATE SYSTEMATIC FOLDERS
        self.aligned_dir = os.path.join(self.base_dir, 'aligned')
        self.sensor_dir = os.path.join(self.base_dir, 'sensors')
        self.planning_dir = os.path.join(self.base_dir, 'planning')
        self.perception_dir = os.path.join(self.base_dir, 'perception')
        
        os.makedirs(self.aligned_dir, exist_ok=True)
        os.makedirs(self.sensor_dir, exist_ok=True)
        os.makedirs(self.planning_dir, exist_ok=True)
        for cam in CAM_CAPS:
            os.makedirs(os.path.join(self.perception_dir, cam), exist_ok=True)
        
        # Perception Tracking
        self.stats = {cam: {'rmse': [], 'd1': [], 'miou': []} for cam in CAM_CAPS.keys()}
        self.frame_counts = {cam: 0 for cam in CAM_CAPS.keys()}
        self.last_timestamps = {cam: None for cam in CAM_CAPS.keys()}
        self.fps_records = {cam: collections.deque(maxlen=30) for cam in CAM_CAPS.keys()}
        self.perception_manifest_buffer = []
        
        # Trajectory Buffers
        self.gt_data = []
        self.ekf_data = []
        self.vo_data = []
        self.slam_data = []
        self.global_path_data = []
        self.local_path_data = []
        self.vgp_data = []
        self.current_gt_time = 0.0
        
        # Raw Sensor Buffers (Saved unconditionally)
        self.raw_imu_data = []
        self.raw_mag_data = []      # Magnetometer Buffer
        self.raw_alt_data = []
        self.raw_wheel_data = []    # Wheel + Steering Buffer
        self.raw_gnss_data = []
        
        self.eval_cache = {cam: collections.OrderedDict() for cam in CAM_CAPS.keys()}
        self.latest_gt = {cam: {'gt_d': None, 'gt_s': None} for cam in CAM_CAPS.keys()}

        # Perception Subscriptions
        for cam in CAM_CAPS.keys():
            self.create_subscription(Image, f'/inference/{cam}/depth', lambda msg, c=cam: self.cache_callback(msg, c, 'd', False), qos_profile_sensor_data, callback_group=self.callback_group)
            self.create_subscription(Image, f'/inference/{cam}/seg', lambda msg, c=cam: self.cache_callback(msg, c, 's', False), qos_profile_sensor_data, callback_group=self.callback_group)
            if self.debug:
                self.create_subscription(Image, f'/carla/hero/depth_{cam}/image', lambda msg, c=cam: self.cache_callback(msg, c, 'd', True), qos_profile_sensor_data, callback_group=self.callback_group)
                self.create_subscription(Image, f'/carla/hero/seg_{cam}/image', lambda msg, c=cam: self.cache_callback(msg, c, 's', True), qos_profile_sensor_data, callback_group=self.callback_group)
        
        # Odometry Subscriptions
        self.create_subscription(Odometry, '/carla/hero/odometry', self.gt_odom_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        self.create_subscription(Odometry, '/ekf/odometry', self.ekf_odom_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        self.create_subscription(Odometry, '/vo/odometry', self.vo_odom_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        self.create_subscription(Odometry, '/slam/odometry', self.slam_odom_callback, qos_profile_sensor_data, callback_group=self.callback_group)

        # Planning Subscriptions
        self.create_subscription(NavPath, '/planning/global_path', self.global_path_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        self.create_subscription(NavPath, '/planning/local_path', self.local_path_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        self.create_subscription(Float32, '/planning/target_speed', self.vgp_callback, 10, callback_group=self.callback_group)

        # Raw Sensor Subscriptions
        self.create_subscription(Imu, '/carla/hero/imu', self.tune_imu_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        self.create_subscription(Imu, '/carla/hero/magnetometer', self.tune_mag_callback, qos_profile_sensor_data, callback_group=self.callback_group) 
        self.create_subscription(PointStamped, '/carla/altimeter', self.tune_alt_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        self.create_subscription(Odometry, '/carla/hero/wheel_odom', self.tune_wheel_callback, qos_profile_sensor_data, callback_group=self.callback_group)
        
        self.gnss_front_sub = message_filters.Subscriber(self, NavSatFix, '/carla/hero/gnss_front')
        self.gnss_rear_sub = message_filters.Subscriber(self, NavSatFix, '/carla/hero/gnss_rear')
        self.gnss_sync = message_filters.ApproximateTimeSynchronizer([self.gnss_front_sub, self.gnss_rear_sub], queue_size=10, slop=0.05)
        self.gnss_sync.registerCallback(self.tune_gnss_callback)

        self.get_logger().info(f"🧪 Unified Evaluation Node Started | Debug: {self.debug} | Tune: {self.tune}")
        self.get_logger().info("✅ Raw Sensor Data (GNSS, IMU, Mag, Wheel) enabled for baseline logging.")

    # ==========================================
    # PATH PLANNING & VELOCITY CALLBACKS
    # ==========================================
    def vgp_callback(self, msg: Float32):
        if self.current_gt_time == 0.0: return
        self.vgp_data.append({'Timestamp': self.current_gt_time, 'Target_Speed': msg.data})

    def global_path_callback(self, msg: NavPath):
        pts = [{'X': p.pose.position.x, 'Y': p.pose.position.y, 'Z': p.pose.position.z} for p in msg.poses]
        if len(pts) > 0 and pts != self.global_path_data:
            self.global_path_data = pts

    def local_path_callback(self, msg: NavPath):
        if self.current_gt_time == 0.0: return
        path_pts = [{'X': p.pose.position.x, 'Y': p.pose.position.y, 'Z': p.pose.position.z} for p in msg.poses]
        self.local_path_data.append([self.current_gt_time, json.dumps(path_pts, separators=(',',':'))])

    # ==========================================
    # TUNING DATA COLLECTION CALLBACKS
    # ==========================================
    def tune_gnss_callback(self, front_msg: NavSatFix, rear_msg: NavSatFix):
        if self.current_gt_time == 0.0: return
        t = front_msg.header.stamp.sec + front_msg.header.stamp.nanosec * 1e-9
        self.raw_gnss_data.append({
            'Timestamp': t,
            'Front_Lat': front_msg.latitude, 'Front_Lon': front_msg.longitude, 'Front_Alt': front_msg.altitude,
            'Rear_Lat': rear_msg.latitude, 'Rear_Lon': rear_msg.longitude, 'Rear_Alt': rear_msg.altitude
        })

    def tune_imu_callback(self, msg: Imu):
        if self.current_gt_time == 0.0: return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.raw_imu_data.append({
            'Timestamp': t, 
            'Accel_X': msg.linear_acceleration.x, 'Accel_Y': msg.linear_acceleration.y, 'Accel_Z': msg.linear_acceleration.z,
            'Gyro_X': msg.angular_velocity.x, 'Gyro_Y': msg.angular_velocity.y, 'Gyro_Z': msg.angular_velocity.z,
            'Qw': msg.orientation.w, 'Qx': msg.orientation.x, 'Qy': msg.orientation.y, 'Qz': msg.orientation.z
        })

    def tune_mag_callback(self, msg: Imu):
        if self.current_gt_time == 0.0: return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.raw_mag_data.append({
            'Timestamp': t, 
            'Qw': msg.orientation.w, 'Qx': msg.orientation.x, 'Qy': msg.orientation.y, 'Qz': msg.orientation.z
        })

    def tune_alt_callback(self, msg: PointStamped):
        if self.current_gt_time == 0.0: return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.raw_alt_data.append({'Timestamp': t, 'Alt_Z': msg.point.z})

    def tune_wheel_callback(self, msg: Odometry):
        if self.current_gt_time == 0.0: return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.raw_wheel_data.append({
            'Timestamp': t, 
            'Odom_Velocity': msg.twist.twist.linear.x,
            'Vx': msg.twist.twist.linear.x,
            'Vy': msg.twist.twist.linear.y,
            'Vz': msg.twist.twist.linear.z,
            'Steering_Angle': msg.twist.twist.angular.z 
        })

    # ==========================================
    # ASYNC STATE-BASED CACHE LOGIC
    # ==========================================
    def cache_callback(self, msg, cam, modality, is_gt):
        if self.current_gt_time == 0.0: return
        
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
                    self.process_evaluation(cam, frame_data, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
                    if t_key in self.eval_cache[cam]: del self.eval_cache[cam][t_key]
            else:
                self.process_evaluation(cam, frame_data, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
                if t_key in self.eval_cache[cam]: del self.eval_cache[cam][t_key]

        if len(self.eval_cache[cam]) > 5:
            self.eval_cache[cam].popitem(last=False)

    def process_evaluation(self, cam_name, frame_data, timestamp):
        try:
            now = time.perf_counter()
            if self.last_timestamps[cam_name] is not None:
                dt = now - self.last_timestamps[cam_name]
                if dt > 0: self.fps_records[cam_name].append(1.0 / dt)
            self.last_timestamps[cam_name] = now
            self.frame_counts[cam_name] += 1

            if not self.debug: return

            pred_d = self.bridge.imgmsg_to_cv2(frame_data['pred_d'], desired_encoding="32FC1")
            pred_s = self.bridge.imgmsg_to_cv2(frame_data['pred_s'], desired_encoding="8UC1")
            
            # Helper decode inline
            gt_d_msg = frame_data['gt_d']
            if gt_d_msg.encoding in ('bgra8','rgba8'):
                img = self.bridge.imgmsg_to_cv2(gt_d_msg, desired_encoding='passthrough')
                normalized = (img[:,:,2].astype(np.float64) + img[:,:,1].astype(np.float64)*256.0 + img[:,:,0].astype(np.float64)*256.0*256.0) / (256.0**3 - 1.0)
                gt_d = (normalized * 1000.0).astype(np.float32)
            else:
                gt_d = self.bridge.imgmsg_to_cv2(gt_d_msg, desired_encoding='32FC1')

            gt_s_msg = frame_data['gt_s']
            if gt_s_msg.encoding in ('bgra8','rgba8'):
                gt_s = self.bridge.imgmsg_to_cv2(gt_s_msg, desired_encoding='passthrough')[:,:,2].astype(np.uint8)
            else:
                gt_s = self.bridge.imgmsg_to_cv2(gt_s_msg, desired_encoding='8UC1')

            if pred_d.shape != gt_d.shape: pred_d = cv2.resize(pred_d, (gt_d.shape[1], gt_d.shape[0]), interpolation=cv2.INTER_LINEAR)
            if pred_s.shape != gt_s.shape: pred_s = cv2.resize(pred_s, (gt_s.shape[1], gt_s.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Dump npz for metric_viz to process
            ts_key = f'{timestamp:.9f}'.replace('.','_')
            out_path = os.path.join(self.perception_dir, cam_name, f'frame_{ts_key}.npz')
            np.savez_compressed(out_path, pred_depth=pred_d, pred_seg=pred_s, gt_depth=gt_d, gt_seg=gt_s)
            self.perception_manifest_buffer.append([cam_name, timestamp, os.path.relpath(out_path, self.base_dir), pred_d.shape[1], pred_d.shape[0]])

        except Exception as e:
            self.get_logger().error(f"Evaluation failed for {cam_name}: {e}")

    # ==========================================
    # ODOMETRY DATA GATHERING 
    # ==========================================
    def gt_odom_callback(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.current_gt_time = t 
        pos, q, v = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear
        yaw, pitch, roll = quaternion_to_euler(q.w, q.x, q.y, q.z)
        self.gt_data.append({
            'Timestamp': t, 'X': pos.x, 'Y': pos.y, 'Z': pos.z, 
            'Yaw': yaw, 'Pitch': pitch, 'Roll': roll, 
            'Velocity': math.sqrt(v.x**2 + v.y**2 + v.z**2),
            'Vx': v.x, 'Vy': v.y, 'Vz': v.z
        })

    def ekf_odom_callback(self, msg: Odometry):
        if self.current_gt_time == 0.0: return
        pos, q, v = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear
        yaw, pitch, roll = quaternion_to_euler(q.w, q.x, q.y, q.z)
        self.ekf_data.append({
            'Timestamp': self.current_gt_time, 'X': pos.x, 'Y': pos.y, 'Z': pos.z, 
            'Yaw': yaw, 'Pitch': pitch, 'Roll': roll, 
            'Vx': v.x, 'Vy': v.y, 'Vz': v.z,
            'P_X': msg.pose.covariance[0] if len(msg.pose.covariance) == 36 else 0.0, 
            'P_Y': msg.pose.covariance[7] if len(msg.pose.covariance) == 36 else 0.0, 
            'Bias_Gx': msg.twist.twist.angular.x, 'Bias_Gy': msg.twist.twist.angular.y, 'Bias_Gz': msg.twist.twist.angular.z
        })

    def slam_odom_callback(self, msg: Odometry):
        if self.current_gt_time == 0.0: return
        yaw, _, _ = quaternion_to_euler(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        self.slam_data.append({'Timestamp': self.current_gt_time, 'X': msg.pose.pose.position.x, 'Y': msg.pose.pose.position.y, 'Z': msg.pose.pose.position.z, 'Yaw': yaw})

    def vo_odom_callback(self, msg: Odometry):
        if self.current_gt_time == 0.0: return
        self.vo_data.append({'Timestamp': self.current_gt_time, 'X': msg.pose.pose.position.x, 'Y': msg.pose.pose.position.y, 'Z': msg.pose.pose.position.z})

    # ==========================================
    # SHUTDOWN ROUTINE (DATA ALIGNMENT & DUMPING)
    # ==========================================
    def align_and_dump_data(self):
        # 1. ALIGN ODOMETRY & SAVE
        if self.gt_data:
            gt_df = pd.DataFrame(self.gt_data).sort_values('Timestamp')
            
            # Align EKF
            if self.ekf_data:
                ekf_df = pd.DataFrame(self.ekf_data).sort_values('Timestamp')
                aligned = pd.merge_asof(ekf_df, gt_df, on='Timestamp', direction='nearest', tolerance=0.05, suffixes=('_EKF', '_GT')).dropna(subset=['X_GT'])
                rows = []
                for _, r in aligned.iterrows():
                    rows.append([
                        r['Timestamp'], r['Timestamp'], 
                        r['X_GT'], r['Y_GT'], r['Z_GT'], r['X_EKF'], r['Y_EKF'], r['Z_EKF'], 
                        r['Yaw_GT'], r['Yaw_EKF'], r['Pitch_GT'], r['Pitch_EKF'], r['Roll_GT'], r['Roll_EKF'], 
                        r.get('Velocity', 0.0), r.get('Vx_GT', 0.0), r.get('Vy_GT', 0.0), r.get('Vz_GT', 0.0), 
                        r.get('Vx_EKF', 0.0), r.get('Vy_EKF', 0.0), r.get('Vz_EKF', 0.0), 
                        r.get('P_X', 0.0), r.get('P_Y', 0.0), 
                        r.get('Bias_Gx', 0.0), r.get('Bias_Gy', 0.0), r.get('Bias_Gz', 0.0), 0.0
                    ])
                with open(os.path.join(self.aligned_dir, 'ekf_aligned.csv'), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'GT_Timestamp','EKF_Timestamp',
                        'GT_X','GT_Y','GT_Z','EKF_X','EKF_Y','EKF_Z',
                        'GT_Yaw','EKF_Yaw','GT_Pitch','EKF_Pitch','GT_Roll','EKF_Roll',
                        'GT_Velocity','GT_Vx','GT_Vy','GT_Vz',
                        'EKF_Vx','EKF_Vy','EKF_Vz',
                        'P_X','P_Y','Bias_Gx','Bias_Gy','Bias_Gz','Time_Diff'
                    ])
                    writer.writerows(rows)
            
            # Align SLAM
            if self.slam_data:
                slam_df = pd.DataFrame(self.slam_data).sort_values('Timestamp')
                aligned = pd.merge_asof(slam_df, gt_df, on='Timestamp', direction='forward', tolerance=0.05, suffixes=('_SLAM', '_GT')).dropna(subset=['X_GT'])
                rows = []
                for _, r in aligned.iterrows():
                    rows.append([r['Timestamp'], r['Timestamp'], r['X_GT'], r['Y_GT'], r['Z_GT'], r['X_SLAM'], r['Y_SLAM'], r['Z_SLAM'], r['Yaw_GT'], r['Yaw_SLAM'], 0.0])
                with open(os.path.join(self.aligned_dir, 'slam_aligned.csv'), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['GT_Timestamp','SLAM_Timestamp','GT_X','GT_Y','GT_Z','SLAM_X','SLAM_Y','SLAM_Z','GT_Yaw','SLAM_Yaw','Time_Diff'])
                    writer.writerows(rows)
            
            # Align VO
            if self.vo_data:
                vo_df = pd.DataFrame(self.vo_data).sort_values('Timestamp')
                aligned = pd.merge_asof(vo_df, gt_df, on='Timestamp', direction='nearest', tolerance=0.05, suffixes=('_VO', '_GT')).dropna(subset=['X_GT'])
                rows = []
                for _, r in aligned.iterrows():
                    rows.append([r['Timestamp'], r['Timestamp'], r['X_GT'], r['Y_GT'], r['Z_GT'], r['X_VO'], r['Y_VO'], r['Z_VO'], 0.0])
                with open(os.path.join(self.aligned_dir, 'vo_aligned.csv'), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['GT_Timestamp','VO_Timestamp','GT_X','GT_Y','GT_Z','VO_X','VO_Y','VO_Z','Time_Diff'])
                    writer.writerows(rows)

        # 2. SAVE RAW SENSORS
        if self.raw_imu_data: pd.DataFrame(self.raw_imu_data).to_csv(os.path.join(self.sensor_dir, 'imu.csv'), index=False)
        if self.raw_mag_data: pd.DataFrame(self.raw_mag_data).to_csv(os.path.join(self.sensor_dir, 'magnetometer.csv'), index=False)
        if self.raw_alt_data: pd.DataFrame(self.raw_alt_data).to_csv(os.path.join(self.sensor_dir, 'altimeter.csv'), index=False)
        if self.raw_wheel_data: pd.DataFrame(self.raw_wheel_data).to_csv(os.path.join(self.sensor_dir, 'wheel_odom.csv'), index=False)
        if self.raw_gnss_data: pd.DataFrame(self.raw_gnss_data).to_csv(os.path.join(self.sensor_dir, 'gnss.csv'), index=False)
        
        # 3. SAVE PLANNING & PERCEPTION
        if self.global_path_data:
            with open(os.path.join(self.planning_dir, 'global_path.json'), 'w') as f: json.dump(self.global_path_data, f)
        if self.local_path_data:
            with open(os.path.join(self.planning_dir, 'local_path.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Path_JSON'])
                writer.writerows(self.local_path_data)
        if self.vgp_data: pd.DataFrame(self.vgp_data).to_csv(os.path.join(self.planning_dir, 'target_speed.csv'), index=False)
        if self.perception_manifest_buffer:
            with open(os.path.join(self.perception_dir, 'manifest.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Camera', 'Timestamp', 'File', 'Width', 'Height'])
                writer.writerows(self.perception_manifest_buffer)

    def destroy_node(self):
        self.align_and_dump_data()
        super().destroy_node()

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
        # 1. Safely wipe node resources and dump the synchronized CSVs
        node.destroy_node()
        
        debug_flag = "--debug" if node.debug else ""
        
        # 2. Asynchronously fire plotting scripts. This completely bypasses ROS 2's GIL lock,
        # stopping the script from hanging and crashing with SIGKILL.
        subprocess.Popen(f"python3 src/autonomy_evaluation/autonomy_evaluation/metric_viz.py --run-dir {node.base_dir} {debug_flag}", shell=True)
        
        executor.shutdown(timeout_sec=0.1)
        if rclpy.ok():
            rclpy.shutdown()
            
        # 3. Instantly kill the process tree to free up the terminal immediately
        os._exit(0) 

if __name__ == '__main__':
    main()