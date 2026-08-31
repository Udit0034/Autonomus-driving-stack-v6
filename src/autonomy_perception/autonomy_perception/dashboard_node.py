#!/usr/bin/env python3
import numpy as np
import cv2
import json
import collections
import threading

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from nav_msgs.msg import OccupancyGrid, Path
from cv_bridge import CvBridge

# ==========================================================
# CONFIGURATION
# ==========================================================

CITYSCAPES_PALETTE = np.array([
    (0, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
    (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230),
    (119, 11, 32), (110, 190, 160), (170, 120, 50), (55, 90, 80), (45, 60, 150), (157, 234, 50),
    (81, 0, 81), (150, 100, 100), (230, 150, 140), (180, 165, 180), (250, 170, 160)
], dtype=np.uint8)

class DashboardNode(Node):
    def __init__(self):
        super().__init__('dashboard_node')
        self.bridge = CvBridge()
        self.callback_group = ReentrantCallbackGroup()
        
        # Shared states
        self.image_cache = {} 
        self.cache_keys = collections.deque(maxlen=30)
        self.cache_lock = threading.Lock()
        self.latest_sim_stamp = None
        
        self.local_path = [] 
        self.global_path = []
        self.lane_offset = 0.0

        # Unified Perception State
        self.latest_signs = []
        self.signs_lock = threading.Lock()

        # Lazy Init Flags
        self._shared_image_sub_active = False
        self._shared_path_sub_active = False

        self.get_logger().info(f"📊 Modular Visual Dashboard Live | Compute-on-Demand Active.")

    # ==========================================================
    # LAZY INITIALIZATION HELPERS
    # ==========================================================
    def _setup_shared_image_cache(self):
        if not self._shared_image_sub_active:
            self.create_subscription(Image, '/carla/hero/front_left/image', self.raw_image_callback, 10, callback_group=self.callback_group)
            self._shared_image_sub_active = True

    def _setup_shared_path_data(self):
        if not self._shared_path_sub_active:
            self.create_subscription(Path, '/planning/local_path', self.local_path_callback, 10, callback_group=self.callback_group)
            self.create_subscription(Path, '/planning/global_path', self.global_path_callback, 10, callback_group=self.callback_group)
            self.create_subscription(Float32, '/perception/lane_center_offset', self.lane_callback, 10, callback_group=self.callback_group)
            self._shared_path_sub_active = True

    # ==========================================================
    # TOGGLEABLE ENABLE METHODS
    # ==========================================================
    def enable_pred_bev(self):
        self.get_logger().info("✅ Enabled: BEV Prediction (GPU Mono Accelerated)")
        self.pub_pred_bev_viz = self.create_publisher(Image, '/dashboard/pred/bev_viz', 10)
        self.create_subscription(Image, '/dashboard/pred/bev', self.pred_bev_callback, 10, callback_group=self.callback_group)

    def enable_perception_overlay(self):
        self.get_logger().info("✅ Enabled: Unified Perception Overlay (YOLO + Signs)")
        self._setup_shared_image_cache()
        self.pub_perception_overlay = self.create_publisher(Image, '/dashboard/perception_overlay', 10)
        
        # Subscribe to both, but render together in the YOLO callback
        self.create_subscription(String, '/perception/sign_detections', self.sign_json_callback, 10, callback_group=self.callback_group)
        self.create_subscription(String, '/inference/front_left/objects', self.yolo_json_callback, 10, callback_group=self.callback_group)

    def enable_ttc_radar(self):
        self.get_logger().info("✅ Enabled: TTC Radar")
        self.pub_ttc_radar = self.create_publisher(Image, '/dashboard/planning/ttc_radar', 10)
        self.create_subscription(String, '/planning/fused_tracked_objects', self.ttc_radar_callback, 10, callback_group=self.callback_group)

    def enable_lidar_occupancy_grid(self):
        self.get_logger().info("✅ Enabled: Lidar Occupancy Grid")
        self._setup_shared_path_data()
        self.pub_lidar_occ = self.create_publisher(Image, '/dashboard/slam/lidar_occupancy', 10) 
        self.create_subscription(OccupancyGrid, '/slam/local_costmap', self.lidar_occupancy_callback, 10, callback_group=self.callback_group)

    def enable_semantic_occupancy_grid(self):
        self.get_logger().info("✅ Enabled: Semantic Occupancy Grid")
        self._setup_shared_path_data()
        self.pub_sem_occ = self.create_publisher(Image, '/dashboard/perception/semantic_occupancy', 10)
        self.create_subscription(OccupancyGrid, '/perception/semantic_costmap', self.semantic_occupancy_callback, 10, callback_group=self.callback_group)

    # ==========================================================
    # DATA INGESTION CALLBACKS
    # ==========================================================
    def raw_image_callback(self, msg):
        self.latest_sim_stamp = msg.header.stamp
        timestamp = f"{msg.header.stamp.sec}_{msg.header.stamp.nanosec}"
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        with self.cache_lock:
            if timestamp not in self.image_cache:
                self.cache_keys.append(timestamp)
            self.image_cache[timestamp] = cv_img
            while len(self.image_cache) > 30:
                oldest = self.cache_keys.popleft()
                self.image_cache.pop(oldest, None)

    def local_path_callback(self, msg: Path):
        self.local_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def global_path_callback(self, msg: Path):
        self.global_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def lane_callback(self, msg: Float32):
        self.lane_offset = msg.data

    # ==========================================================
    # RENDERING CALLBACKS 
    # ==========================================================
    def sign_json_callback(self, msg):
        """Silently cache the latest signs to be drawn by the unified overlay."""
        try:
            payload = json.loads(msg.data)
            with self.signs_lock:
                self.latest_signs = payload if payload else []
        except Exception:
            pass

    def yolo_json_callback(self, msg):
        """Master rendering loop for the unified perception overlay."""
        try:
            payload = json.loads(msg.data)
            stamp = payload["header"]["stamp"]
            target_time = f"{stamp['sec']}_{stamp['nanosec']}"
            
            with self.cache_lock:
                if target_time not in self.image_cache:
                    if not self.cache_keys: return
                    target_time = self.cache_keys[-1]
                canvas = self.image_cache[target_time].copy()
            
            # 1. Draw YOLO Detections (Vehicles, Pedestrians, Traffic Lights)
            for det in payload.get("detections", []):
                x1, y1, x2, y2 = det["bbox"]
                cls_name = det["class_name"]
                depth = det["median_depth"]
                tid = det["track_id"]
                color = (255, 255, 0) # Cyan/Yellow
                
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{tid} {cls_name.upper()} | {depth:.1f}m"
                cv2.rectangle(canvas, (x1, max(0, y1-25)), (x1 + len(label)*10, y1), color, -1)
                cv2.putText(canvas, label, (x1+5, max(15, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # 2. Draw Cached Traffic Signs
            with self.signs_lock:
                current_signs = list(self.latest_signs)
                
            for det in current_signs:
                x1, y1, x2, y2 = det["bbox"]
                cls_name = det["class"]
                dist = det["distance"]
                color = (0, 165, 255) # Orange
                
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
                label = f"SIGN: {cls_name.upper()} | {dist:.1f}m"
                cv2.rectangle(canvas, (x1, max(0, y1-25)), (x1 + len(label)*10, y1), color, -1)
                cv2.putText(canvas, label, (x1+5, max(15, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # 3. Memory/Bandwidth Fix: Downsample to half resolution
            canvas = cv2.resize(canvas, (640, 360))
            
            # Publish
            canvas = np.ascontiguousarray(canvas)
            img_msg = self.bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
            if self.latest_sim_stamp is not None: img_msg.header.stamp = self.latest_sim_stamp
            img_msg.header.frame_id = "front_left"
            
            self.pub_perception_overlay.publish(img_msg)
            
        except Exception as e:
            pass

    def ttc_radar_callback(self, msg):
        try:
            targets = json.loads(msg.data)
            res = 0.1 
            size = 400
            canvas = np.zeros((size, size, 3), dtype=np.uint8)
            cx, cy = size // 2, size // 2
            
            cv2.circle(canvas, (cx, cy), int(10/res), (50, 50, 50), 1)
            cv2.circle(canvas, (cx, cy), int(20/res), (50, 50, 50), 1)
            cv2.rectangle(canvas, (cx - 4, cy - 10), (cx + 4, cy + 10), (255, 255, 255), -1)
            
            for obj in targets:
                x_metric, y_metric, spd, ttc, tid, cls_name = obj
                
                px = cx - int(y_metric / res)
                py = cy - int(x_metric / res)
                
                if px < 0 or px >= size or py < 0 or py >= size: continue
                    
                if ttc < 3.0: color = (0, 0, 255)        
                elif ttc < 6.0: color = (0, 200, 255)    
                else: color = (0, 255, 0)                
                    
                cv2.circle(canvas, (px, py), 6, color, -1)
                label = f"{cls_name[:3].upper()} {ttc:.1f}s"
                cv2.putText(canvas, label, (px + 10, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
            cv2.putText(canvas, "FUSED TTC RADAR (40m)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            canvas = np.ascontiguousarray(canvas)
            img_msg = self.bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
            if self.latest_sim_stamp is not None: img_msg.header.stamp = self.latest_sim_stamp
            img_msg.header.frame_id = "base_link"
            self.pub_ttc_radar.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"TTC Radar Render Error: {e}")

    def draw_paths_and_lane_center(self, canvas, cx, cy, res):
        offset_px = int(cx - (self.lane_offset / res))
        cv2.line(canvas, (offset_px, 0), (offset_px, canvas.shape[0]), (0, 255, 255), 2)

        if self.local_path:
            for i in range(len(self.local_path) - 1):
                x1, y1 = self.local_path[i]
                x2, y2 = self.local_path[i+1]
                px1, py1 = int(cx - (y1 / res)), int(cy - (x1 / res))
                px2, py2 = int(cx - (y2 / res)), int(cy - (x2 / res))
                cv2.line(canvas, (px1, py1), (px2, py2), (255, 0, 255), 2)

    def lidar_occupancy_callback(self, msg: OccupancyGrid):
        try:
            # 1. Parse grid and flip Y so it matches OpenCV coordinate system visually
            grid_array = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
            grid_array = np.flipud(grid_array) 
            
            res = msg.info.resolution
            ox = msg.info.origin.position.x
            oy = msg.info.origin.position.y
            
            # Create colored canvas
            canvas = np.zeros((msg.info.height, msg.info.width, 3), dtype=np.uint8)
            canvas[grid_array == -1] = [20, 20, 25]     # Unknown (Dark Gray)
            canvas[grid_array == 0] = [0, 0, 0]         # Free Space
            canvas[grid_array >= 50] = [0, 255, 150]    # Occupied (Neon Green)

            # 2. Determine Ego Position and Heading (Yaw) from the Path
            if len(self.local_path) > 1:
                ego_wx, ego_wy = self.local_path[0]
                next_wx, next_wy = self.local_path[1]
                yaw_rad = np.arctan2(next_wy - ego_wy, next_wx - ego_wx)
            else:
                # Fallback if no path is published
                ego_wx = ox + (msg.info.width / 2.0) * res
                ego_wy = oy + (msg.info.height / 2.0) * res
                q = msg.info.origin.orientation
                yaw_rad = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

            # Calculate precise pixel coordinates of the Ego vehicle on the raw canvas
            ego_px = int((ego_wx - ox) / res)
            ego_py = int(msg.info.height - ((ego_wy - oy) / res))

            # 3. Draw Local Path (Magenta) - No Blue Line
            if self.local_path:
                for i in range(len(self.local_path) - 1):
                    x1, y1 = self.local_path[i]
                    x2, y2 = self.local_path[i+1]
                    
                    px1 = int((x1 - ox) / res)
                    py1 = int(msg.info.height - ((y1 - oy) / res))
                    px2 = int((x2 - ox) / res)
                    py2 = int(msg.info.height - ((y2 - oy) / res))
                    
                    cv2.line(canvas, (px1, py1), (px2, py2), (255, 0, 255), 2)

            # 4. Egocentric World Rotation (The Magic Fix)
            # Find the rotation required to make the car's heading point straight UP (-90 degrees in OpenCV)
            angle_cv = -np.degrees(yaw_rad)
            R = -90 - angle_cv 
            
            # Target output size
            out_size = 600
            cx_out = out_size // 2
            cy_out = out_size // 2
            
            # Get rotation matrix around the car's current pixel, then translate it to the center of the 600x600 window
            M = cv2.getRotationMatrix2D((ego_px, ego_py), R, 1.0)
            M[0, 2] += (cx_out - ego_px)
            M[1, 2] += (cy_out - ego_py)
            
            # Warp the image (rotates the world and centers the car in one step)
            canvas_centered = cv2.warpAffine(canvas, M, (out_size, out_size), borderValue=(20, 20, 25))

            # 5. Draw Ego Car (Now Guaranteed to always be exactly in the center and perfectly Vertical | )
            cv2.rectangle(canvas_centered, (cx_out - 4, cy_out - 10), (cx_out + 4, cy_out + 10), (255, 255, 255), -1)

            # 6. Add Text Overlay
            cv2.putText(canvas_centered, "LIDAR OCCUPANCY (EGO-CENTRIC)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas_centered, f"OFFSET: {self.lane_offset:.2f}m", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 7. Publish
            canvas_centered = np.ascontiguousarray(canvas_centered)
            img_msg = self.bridge.cv2_to_imgmsg(canvas_centered, encoding="bgr8")
            img_msg.header.stamp = msg.header.stamp
            img_msg.header.frame_id = "ego_vehicle"
            self.pub_lidar_occ.publish(img_msg)
            
        except Exception as e: 
            self.get_logger().error(f"Lidar Dashboard Error: {e}")

    def semantic_occupancy_callback(self, msg: OccupancyGrid):
        try:
            grid_array = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
            res = msg.info.resolution
            
            canvas = np.zeros((msg.info.height, msg.info.width, 3), dtype=np.uint8)
            
            canvas[grid_array == -1] = [20, 20, 25]     
            canvas[grid_array == 0] = [0, 255, 0]       
            canvas[grid_array == 60] = [100, 255, 100]  
            canvas[grid_array >= 100] = [0, 0, 255]     
            
            cy, cx = msg.info.height // 2, msg.info.width // 2
            self.draw_paths_and_lane_center(canvas, cx, cy, res)
            cv2.rectangle(canvas, (cx - 4, cy - 10), (cx + 4, cy + 10), (255, 255, 255), -1)
            
            canvas = cv2.resize(canvas, (600, 600), interpolation=cv2.INTER_NEAREST)
            cv2.putText(canvas, "SEMANTIC OCCUPANCY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(canvas, f"OFFSET: {self.lane_offset:.2f}m", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            canvas = np.ascontiguousarray(canvas)
            img_msg = self.bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
            img_msg.header.stamp = msg.header.stamp
            img_msg.header.frame_id = "ego_vehicle"
            self.pub_sem_occ.publish(img_msg)
        except Exception as e: pass

    def pred_bev_callback(self, msg):
        try:
            labels = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            labels_clipped = np.clip(labels, 0, len(CITYSCAPES_PALETTE)-1)
            bev_rgb = CITYSCAPES_PALETTE[labels_clipped]
            
            target_size = (720, 720)
            bev_bgr = cv2.resize(bev_rgb, target_size, interpolation=cv2.INTER_NEAREST)
            bev_bgr = cv2.cvtColor(bev_bgr, cv2.COLOR_RGB2BGR)
            
            cx = target_size[0] // 2
            cy = int(target_size[1] * (50.0 / 80.0))
            
            car_w = 12
            car_l = 24
            cv2.rectangle(bev_bgr, (cx - car_w, cy - car_l), (cx + car_w, cy + car_l), (0, 255, 0), -1)
            cv2.putText(bev_bgr, "PRED: GPU Mono BEV", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            bev_bgr = np.ascontiguousarray(bev_bgr)
            img_msg = self.bridge.cv2_to_imgmsg(bev_bgr, encoding="bgr8")
            img_msg.header = msg.header
            self.pub_pred_bev_viz.publish(img_msg)
            
        except Exception as e:
            self.get_logger().error(f"BEV Viz mapping failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()

    # ==========================================================
    # 🎯 TOGGLE VISUALIZATIONS HERE
    # ==========================================================
    
    # node.enable_pred_bev()
    node.enable_lidar_occupancy_grid()
    node.enable_semantic_occupancy_grid()
    # node.enable_perception_overlay()
    node.enable_ttc_radar()

    try: 
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt: 
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()