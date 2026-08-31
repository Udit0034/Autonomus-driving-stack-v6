import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
from collections import deque
import threading

# ==================================================
# CLASSIC CV: HSV PREDICTOR (From your tuning!)
# ==================================================
HSV_RANGES = {
    "red1": (np.array([0, 100, 100]), np.array([10, 255, 255])),
    "red2": (np.array([160, 100, 100]), np.array([180, 255, 255])),
    "yellow": (np.array([15, 80, 100]), np.array([35, 255, 255])),
    "green": (np.array([40, 80, 80]), np.array([90, 255, 255]))
}

def predict_light_color(crop):
    if crop.size == 0: return "UNKNOWN"
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = crop.shape[0]
    
    mask_r1 = cv2.inRange(hsv, HSV_RANGES["red1"][0], HSV_RANGES["red1"][1])
    mask_r2 = cv2.inRange(hsv, HSV_RANGES["red2"][0], HSV_RANGES["red2"][1])
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    mask_yellow = cv2.inRange(hsv, HSV_RANGES["yellow"][0], HSV_RANGES["yellow"][1])
    mask_green = cv2.inRange(hsv, HSV_RANGES["green"][0], HSV_RANGES["green"][1])

    top, middle, bottom = slice(0, h // 3), slice(h // 3, 2 * h // 3), slice(2 * h // 3, h)

    scores = {
        "RED": np.sum(mask_red[top, :] > 0),
        "YELLOW": np.sum(mask_yellow[middle, :] > 0),
        "GREEN": np.sum(mask_green[bottom, :] > 0)
    }
    
    if sum(scores.values()) < 10: # Minimum active pixel threshold
        return "UNKNOWN"
        
    return max(scores, key=scores.get)

# ==================================================
# ROS2 FUSION NODE
# ==================================================
class TrafficLightFusionNode(Node):
    def __init__(self):
        super().__init__('traffic_light_fusion_node')
        self.bridge = CvBridge()
        
        # 1. Image Cache for Syncing
        self.image_cache = {} 
        self.cache_keys = deque(maxlen=30)
        self.cache_lock = threading.Lock()
        
        # 2. Hysteresis (Temporal Tracking) Variables
        self.confirmed_state = "UNKNOWN"
        self.candidate_state = "UNKNOWN"
        self.consecutive_frames = 0
        self.FRAMES_TO_CONFIRM = 3   # Needs 3 agreeing frames to change state
        self.FRAMES_TO_FORGET = 10   # Holds state for 10 frames if light is temporarily lost
        self.lost_frames_count = 0

        # Subscriptions
        self.create_subscription(Image, '/carla/hero/front_left/image', self.image_callback, 10)
        self.create_subscription(String, '/inference/front_left/traffic_lights', self.json_callback, 10)
        
        # CHANGE 1: Seg map subscription as secondary detector
        self.latest_seg = None
        self.seg_lock = threading.Lock()
        self.create_subscription(Image, '/inference/front_left/seg', self.seg_callback, 10)
        
        # Publisher to the FSM
        self.state_pub = self.create_publisher(String, '/fsm/active_traffic_light', 10)
        
        self.get_logger().info("🚦 Traffic Light Fusion Node Active.")

    def image_callback(self, msg):
        """Stores the raw image in a rolling buffer keyed by timestamp."""
        timestamp = f"{msg.header.stamp.sec}_{msg.header.stamp.nanosec}"
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Lock the critical section
        with self.cache_lock:
            # Prevent duplicate timestamps from entering the deque
            if timestamp not in self.image_cache:
                self.cache_keys.append(timestamp)
                
            self.image_cache[timestamp] = cv_img
            
            # Clean up old images safely
            while len(self.image_cache) > 30:
                oldest = self.cache_keys.popleft()
                self.image_cache.pop(oldest, None) # Safe delete

    def seg_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        with self.seg_lock:
            self.latest_seg = cv_img

    def json_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Failed to decode JSON payload.")
            return
        
        # Attempt to find the matching image in our cache
        stamp = payload["header"]["stamp"]
        target_time = f"{stamp['sec']}_{stamp['nanosec']}"
        
        # Safely extract the frame using the lock
        with self.cache_lock:
            if target_time not in self.image_cache:
                if not self.cache_keys: return
                target_time = self.cache_keys[-1]
            frame = self.image_cache.get(target_time)
            
        # If frame is somehow still None, abort this cycle
        if frame is None:
            return
            
        meta = payload["camera_meta"]
        fx, cx = meta["intrinsics"]["fx"], meta["intrinsics"]["cx"]
        
        valid_candidates = []

        # ==================================================
        # 1. EVALUATE ALL DETECTIONS (YOLO PRIMARY)
        # ==================================================
        for det in payload["detections"]:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            z_depth = det["median_depth"]
            yolo_conf = det["yolo_conf"]
            seg_ratio = det["seg_verification_ratio"]
            
            # --- REDUNDANCY FUSION CHECK ---
            if (yolo_conf < 0.4 and seg_ratio < 0.3):
                continue
            
            # --- 3D SPATIAL GATING ---
            u_center = (x1 + x2) / 2.0
            x_world = (u_center - cx) * (z_depth / fx)
            
            if z_depth > 80.0 or x_world < -2.0 or x_world > 15.0:
                continue
                
            # If it survived, it's a valid light!
            valid_candidates.append({
                "bbox": [x1, y1, x2, y2],
                "depth": z_depth,
                "confidence": (yolo_conf + seg_ratio) / 2.0 # Fused confidence
            })

        # ==================================================
        # 2. SEG-BASED FALLBACK (CHANGE 2)
        # ==================================================
        if not valid_candidates:
            with self.seg_lock:
                seg = self.latest_seg.copy() if self.latest_seg is not None else None
            
            if seg is not None:
                # Look for traffic light pixels (class 6 or 7 in Cityscapes)
                tl_mask = np.isin(seg, [6, 7]).astype(np.uint8) * 255
                contours, _ = cv2.findContours(tl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    if cv2.contourArea(cnt) < 50: continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Estimate depth from bbox size (traffic lights ~0.5m diameter)
                    est_depth = 640.0 * 0.5 / max(w, 1)
                    if est_depth > 80.0: continue
                    
                    # Crop and classify color to verify it's a valid light
                    crop = frame[y:y+h, x:x+w]
                    color = predict_light_color(crop)
                    if color != "UNKNOWN":
                        valid_candidates.append({
                            "bbox": [x, y, x+w, y+h],
                            "depth": est_depth,
                            "confidence": 0.6,
                            "source": "seg"
                        })

        # ==================================================
        # 3. SELECTION & CLASSIFICATION
        # ==================================================
        current_observation = "UNKNOWN"
        best_distance = -1.0 # Initialize the distance so it always exists! (Change 3)
        
        if valid_candidates:
            self.lost_frames_count = 0
            
            # Sort by depth and pick the NEAREST one
            valid_candidates.sort(key=lambda item: item["depth"])
            best_light = valid_candidates[0]
            best_distance = float(best_light["depth"]) # Safely capture distance
            
            x1, y1, x2, y2 = best_light["bbox"]
            crop = frame[y1:y2, x1:x2]
            
            # Run the Classic CV Algorithm
            current_observation = predict_light_color(crop)

        else:
            self.lost_frames_count += 1

        # ==================================================
        # 4. TEMPORAL HYSTERESIS (STATE MACHINE)
        # ==================================================
        
        # If we temporarily lose sight of the light, hold the state for a bit
        if current_observation == "UNKNOWN" and self.lost_frames_count < self.FRAMES_TO_FORGET:
            current_observation = self.confirmed_state
            
        # Update Hysteresis Logic
        if current_observation == self.candidate_state:
            self.consecutive_frames += 1
        else:
            self.candidate_state = current_observation
            self.consecutive_frames = 1
            
        # Lock in the state if confirmed N times
        if self.consecutive_frames >= self.FRAMES_TO_CONFIRM:
            self.confirmed_state = self.candidate_state
            
        # Publish to the vehicle's controller
        output_msg = json.dumps({
            "state": self.confirmed_state,
            "distance_m": best_distance
        })
        self.state_pub.publish(String(data=output_msg))


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass # Gracefully catch the Ctrl+C
    finally:
        # Stop ROS 2 safely
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()