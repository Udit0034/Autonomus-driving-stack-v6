import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Header, String
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import cv2
import numpy as np
import tensorrt as trt
import time
import concurrent.futures
from scipy.spatial import distance
import json
import traceback

import torch
import torch.nn.functional as F

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

# =============================================================================
# CONFIGURATION & CLASSES
# =============================================================================
CLASSES = {
    0: 'Vehicle',
    1: 'Pedestrian',
    2: 'Traffic_Light',
    3: 'Traffic_Sign'
}

SIGN_CLASSES = {
    0: 'speed_30', 
    1: 'speed_60', 
    2: 'speed_90', 
    3: 'stop'
}

CAMS = {
    "front_left": {"fov": 90.0, "w": 1280, "h": 720, "x": 1.4, "y": -0.25, "z": 1.5, "pitch": -11.0, "yaw": 0.0, "roll": 0.0, "max_depth": 250.0},
    "rear":       {"fov": 100.0, "w": 800, "h": 400, "x": -2.0, "y": 0.0, "z": 1.6, "pitch": -26.0, "yaw": 180.0, "roll": 0.0, "max_depth": 150.0},
    "side_left":  {"fov": 120.0, "w": 800, "h": 400, "x": 0.0, "y": -0.8, "z": 1.8, "pitch": -41.0, "yaw": -90.0, "roll": 0.0, "max_depth": 80.0},
    "side_right": {"fov": 120.0, "w": 800, "h": 400, "x": 0.0, "y": 0.8, "z": 1.8, "pitch": -41.0, "yaw": 90.0, "roll": 0.0, "max_depth": 80.0},
}

def make_homogeneous_transform(cam):
    p, y, r = np.deg2rad([cam["pitch"], cam["yaw"], cam["roll"]])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.array([[cam["x"]], [cam["y"]], [cam["z"]]])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3:] = T
    return M.astype(np.float32)

class CentroidTracker:
    def __init__(self, max_distance=80, max_lost=10):
        self.next_id = 1
        self.tracks = {}  
        self.max_distance = max_distance
        self.max_lost = max_lost

    def update(self, detections):
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['lost'] += 1
                if self.tracks[track_id]['lost'] > self.max_lost:
                    del self.tracks[track_id]
            return {}

        input_centroids = np.array([[d[0], d[1]] for d in detections])
        input_classes = [d[2] for d in detections]
        input_boxes = [d[3] for d in detections]

        if len(self.tracks) == 0:
            for i in range(len(input_centroids)):
                self.tracks[self.next_id] = {'centroid': input_centroids[i], 'lost': 0, 'class': input_classes[i]}
                self.next_id += 1
        else:
            track_ids = list(self.tracks.keys())
            track_centroids = np.array([self.tracks[tid]['centroid'] for tid in track_ids])

            D = distance.cdist(track_centroids, input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols: continue
                if D[row, col] > self.max_distance: continue

                if self.tracks[track_ids[row]]['class'] == input_classes[col]:
                    track_id = track_ids[row]
                    self.tracks[track_id]['centroid'] = input_centroids[col]
                    self.tracks[track_id]['lost'] = 0
                    used_rows.add(row)
                    used_cols.add(col)

            unused_cols = set(range(input_centroids.shape[0])) - used_cols
            for col in unused_cols:
                self.tracks[self.next_id] = {'centroid': input_centroids[col], 'lost': 0, 'class': input_classes[col]}
                self.next_id += 1

            unused_rows = set(range(track_centroids.shape[0])) - used_rows
            for row in unused_rows:
                track_id = track_ids[row]
                self.tracks[track_id]['lost'] += 1
                if self.tracks[track_id]['lost'] > self.max_lost:
                    del self.tracks[track_id]

        active_objects = {}
        for i, centroid in enumerate(input_centroids):
            for tid, t_data in self.tracks.items():
                if np.array_equal(t_data['centroid'], centroid) and t_data['lost'] == 0:
                    active_objects[tid] = input_boxes[i]
                    break
        return active_objects

class TRTZeroCopyInstance:
    def __init__(self, engine_path):
        runtime = trt.Runtime(TRT_LOGGER)
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Missing Engine: {engine_path}")
            
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
        self.inputs, self.outputs, self.output_tensors = [], [], {}
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = torch.float16 if trt.nptype(self.engine.get_tensor_dtype(name)) == np.float16 else torch.float32
                self.output_tensors[name] = torch.empty(shape, dtype=dtype, device="cuda").contiguous()

    def enqueue_async(self, *input_tensors):
        for idx, tensor in enumerate(input_tensors):
            self.context.set_tensor_address(self.inputs[idx], tensor.data_ptr())
        for name in self.outputs:
            self.context.set_tensor_address(name, self.output_tensors[name].data_ptr())
            
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        return [self.output_tensors[name] for name in self.outputs]

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')
        self.bridge = CvBridge()
        
        self.declare_parameter('debug_mode', False)
        self.debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value
        
        # Shutdown flag to prevent InvalidHandle crashes
        self.is_shutting_down = False

        self.cpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=12)
        self.engine_dir = "/home/ubuntu/AV6/shared_cache/trt_engine_cache"

        self.mean_gpu = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std_gpu  = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

        self.camera_metadata = {
            "front_left": {
                "fov": 90.0, "w": 1280, "h": 720,
                "x": 1.4, "y": -0.25, "z": 1.5,
                "pitch": -12.0, "yaw": 0.0, "roll": 0.0,
                "intrinsics": {
                    "fx": 640.0, "fy": 640.0, "cx": 640.0, "cy": 360.0
                }
            }
        }

        self.model_manifest = {
            'stereonet': 'baked_stereonet.engine',
            'seg_front': 'front_seg_int8.engine',
            'rear_depth': 'rear_depth_int8.engine',
            'rear_seg': 'rear_seg_int8.engine',
            'side_left_depth': 'side_depth_int8.engine',
            'side_left_seg': 'side_seg_int8.engine',
            'side_right_depth': 'side_depth_int8.engine',
            'side_right_seg': 'side_seg_int8.engine',
            'yolo': 'yolo_int8.engine',
            'sign_classifier': 'carla_sign_int8.engine'
        }

        self.get_logger().info("Loading Zero-Copy TensorRT Engines from cache...")
        self.engines = {}
        for name, file_name in self.model_manifest.items():
            engine_path = os.path.join(self.engine_dir, file_name)
            if os.path.exists(engine_path):
                self.engines[name] = TRTZeroCopyInstance(engine_path)
                self.get_logger().info(f"✅ Loaded engine: {name}")
            else:
                self.get_logger().error(f"❌ Missing engine '{name}'")
        
        self.tracker = CentroidTracker(max_distance=80, max_lost=10)

        self.camera_names = ['front_left', 'front_right', 'rear', 'side_left', 'side_right']
        self.latest_frames = {cam: None for cam in self.camera_names}
        self.frame_counts = {cam: 0 for cam in self.camera_names}
        self.gpu_future = None  
        self.t_prep, self.t_infer, self.t_post, self.t_wall = [], [], [], []
        
        self.init_gpu_bev_projector()

        self.inference_publishers = {cam: {} for cam in self.camera_names}
        for cam in self.camera_names:
            self.create_subscription(Image, f'/carla/hero/{cam}/image', lambda msg, c=cam: self.image_callback(msg, c), qos_profile_sensor_data)
            
            # FIX: Always create front_left Depth & Seg publishers so TTC Fusion and RViz can see them!
            if cam == 'front_left':
                self.inference_publishers[cam]['depth'] = self.create_publisher(Image, f'/inference/{cam}/depth', 10)
                self.inference_publishers[cam]['seg'] = self.create_publisher(Image, f'/inference/{cam}/seg', 10)
            elif self.debug_mode:
                self.inference_publishers[cam]['depth'] = self.create_publisher(Image, f'/inference/{cam}/depth', 10)
                self.inference_publishers[cam]['seg'] = self.create_publisher(Image, f'/inference/{cam}/seg', 10)

        self.bev_pub = self.create_publisher(Image, '/dashboard/pred/bev', 10)
        self.traffic_light_pub = self.create_publisher(String, '/inference/front_left/traffic_lights', 10)
        self.traffic_sign_pub = self.create_publisher(String, '/perception/sign_detections', 10)
        self.dynamic_objects_pub = self.create_publisher(String, '/inference/front_left/objects', 10)
        self.heartbeat_pub = self.create_publisher(Bool, '/inference/heartbeat', 10)

        self.pin_buffers = {}
        for cam in self.camera_names:
            if 'front' in cam:
                self.pin_buffers[cam] = torch.empty((720, 1280, 3), dtype=torch.uint8, pin_memory=True)
            else:
                self.pin_buffers[cam] = torch.empty((400, 800, 3), dtype=torch.uint8, pin_memory=True)
        
        self.timer = self.create_timer(0.01, self.inference_loop)
        self.get_logger().info(f"Native CARLA PyTorch Pipeline Active. Debug: {self.debug_mode}")

    def init_gpu_bev_projector(self):
        self.cam_grids = {}
        self.bev_conf = torch.zeros((800, 500), dtype=torch.float32, device="cuda")
        self.bev_labels = torch.zeros((800, 500), dtype=torch.uint8, device="cuda")
        self.last_bev_time = time.perf_counter()

        for cam, ext in CAMS.items():
            DOWN = 8
            w, h = ext['w'] // DOWN, ext['h'] // DOWN
            ys, xs = torch.meshgrid(torch.arange(h, device="cuda"), torch.arange(w, device="cuda"), indexing='ij')
            
            fx = (ext['w'] / 2) / np.tan(np.deg2rad(ext['fov']) / 2) / float(DOWN)
            ray_x = (xs - (w / 2)) / fx
            ray_y = (ys - (h / 2)) / fx
            
            rays = torch.stack([torch.ones_like(ray_x), ray_x, -ray_y, torch.ones_like(ray_x)], dim=-1).view(-1, 4).T
            
            M = torch.tensor(make_homogeneous_transform(ext), device="cuda")
            rot_rays = M[:3, :3] @ rays[:3, :]
            
            self.cam_grids[cam] = {"rot_rays": rot_rays, "T": M[:3, 3:], "max_depth": ext['max_depth'], "down": DOWN}

    def compute_bev_on_gpu(self, d_maps, s_maps):
        try:
            current_time = time.perf_counter()
            dt = current_time - self.last_bev_time
            self.last_bev_time = current_time
            
            decay_rate = 0.05 * dt
            self.bev_conf -= decay_rate
            self.bev_conf.clamp_(min=0.0, max=1.0)
            
            for cam in CAMS.keys():
                if cam not in d_maps or cam not in s_maps: 
                    continue
                    
                grid = self.cam_grids[cam]
                DOWN = grid["down"]

                d = d_maps[cam][::DOWN, ::DOWN].contiguous().view(-1)
                s = s_maps[cam][::DOWN, ::DOWN].contiguous().view(-1)

                valid = (d > 0.5) & (d < grid['max_depth']) & (s != 0) & (s != 11)
                
                d_v = d[valid]
                s_v = s[valid].to(torch.uint8)
                pts = grid['rot_rays'][:, valid] * d_v + grid['T']
                X, Y = pts[0, :], pts[1, :]
                Z    = pts[2, :]
                
                in_bounds = (X > -30.0) & (X < 50.0) & (Y > -25.0) & (Y < 25.0) & (Z > -5.0) & (Z < 15.0)
                
                X_b = X[in_bounds]
                Y_b = Y[in_bounds]
                s_b = s_v[in_bounds]
                
                rows = torch.clamp((50.0 - X_b) * 10.0, 0, 799).long()
                cols = torch.clamp((Y_b + 25.0) * 10.0, 0, 499).long()

                flat_idx = rows * 500 + cols

                conf_flat = self.bev_conf.view(-1)
                ones = torch.ones(flat_idx.shape[0], dtype=torch.float32, device="cuda")
                conf_flat.scatter_add_(0, flat_idx, ones * 0.5)

                label_flat = self.bev_labels.view(-1)
                label_flat.scatter_(0, flat_idx, s_b)
            
            self.bev_conf.clamp_(max=1.0)
                
            bev = torch.zeros((800, 500), dtype=torch.uint8, device="cuda")
            # FIX: Lowered BEV threshold so individual hits show up instantly
            mask = self.bev_conf >= 0.4
            bev[mask] = self.bev_labels[mask]
            
            bev_resized = F.interpolate(bev.unsqueeze(0).unsqueeze(0).float(), size=(500, 500), mode='nearest').squeeze().to(torch.uint8)
            return bev_resized
            
        except Exception:
            if not self.is_shutting_down:
                self.get_logger().error(f"GPU BEV Compute Error:\n{traceback.format_exc()}")
            return torch.zeros((500, 500), dtype=torch.uint8, device="cuda")

    def image_callback(self, image_msg, cam):
        try:
            bgr_array = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            self.latest_frames[cam] = (image_msg.header, bgr_array)
        except Exception:
            pass

    def preprocess_to_gpu(self, bgr_array, pin_buf):
        pin_buf.copy_(torch.from_numpy(bgr_array))
        raw_hwc_gpu = pin_buf.to(device="cuda", non_blocking=True)
        baked_tensor = raw_hwc_gpu.unsqueeze(0).contiguous()
        
        tensor = raw_hwc_gpu[:, :, [2, 1, 0]]
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
        tensor_float = tensor.float().mul_(1.0 / 255.0)
        tensor_norm = tensor_float.sub(self.mean_gpu).div_(self.std_gpu)
        
        return tensor_norm, tensor_float, baked_tensor

    def publish_bev_worker(self, bev_tensor, header):
        if self.is_shutting_down: return
        try:
            bev_np = np.ascontiguousarray(bev_tensor.cpu().numpy(), dtype=np.uint8)
            img_msg = self.bridge.cv2_to_imgmsg(bev_np, encoding="mono8")
            img_msg.header.stamp = header.stamp
            img_msg.header.frame_id = "base_link"
            self.bev_pub.publish(img_msg)
        except Exception:
            pass

    def process_and_publish_worker(self, cam, depth_tensor, seg_tensor, header):
        if self.is_shutting_down: return
        try:
            if not self.debug_mode and cam != 'front_left': return
            
            if depth_tensor is not None and 'depth' in self.inference_publishers[cam]:
                depth_np = depth_tensor.cpu().numpy()
                depth_msg = self.bridge.cv2_to_imgmsg(depth_np, encoding="32FC1")
                depth_msg.header = header 
                self.inference_publishers[cam]['depth'].publish(depth_msg)
                
            if seg_tensor is not None and 'seg' in self.inference_publishers[cam]:
                seg_np = seg_tensor.cpu().numpy()
                seg_msg = self.bridge.cv2_to_imgmsg(seg_np, encoding="mono8")
                seg_msg.header = header   
                self.inference_publishers[cam]['seg'].publish(seg_msg)
        except Exception:
            pass

    def process_detections_worker(self, yolo_tensor, depth_tensor, seg_tensor, raw_image_tensor, header):
        if self.is_shutting_down or seg_tensor is None:
            return 

        try:
            yolo_np = yolo_tensor.cpu().numpy()
            depth_np = depth_tensor.cpu().numpy() if depth_tensor is not None else None
            seg_np = seg_tensor.cpu().numpy()

            box_data, score_data = yolo_np[:, :4], yolo_np[:, 4:]
            max_scores = np.max(score_data, axis=1)
            
            valid_mask = max_scores > 0.20 
            valid_boxes = box_data[valid_mask]
            valid_scores = max_scores[valid_mask]
            valid_classes = np.argmax(score_data, axis=1)[valid_mask]
            
            boxes, scores, class_ids = [], [], []
            dynamic_detections, static_detections = [], []
            sx, sy = 1280.0 / 640.0, 720.0 / 640.0 

            for i in range(len(valid_boxes)):
                cx, cy, w, h = valid_boxes[i]
                cx *= sx; w *= sx; cy *= sy; h *= sy
                x, y = int(cx - w/2), int(cy - h/2)
                boxes.append([x, y, int(w), int(h)])
                scores.append(float(valid_scores[i]))
                class_ids.append(int(valid_classes[i]))

            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.20, 0.5)
            if len(indices) > 0:
                for i in indices.flatten():
                    cls_id = class_ids[i]
                    box = boxes[i]
                    cx, cy = box[0] + box[2]/2, box[1] + box[3]/2
                    if cls_id in [0, 1]: 
                        dynamic_detections.append([cx, cy, cls_id, box])
                    else:
                        static_detections.append([cls_id, box, scores[i]])

            active_dynamic = self.tracker.update(dynamic_detections)
            h_max, w_max = seg_np.shape[0], seg_np.shape[1] 
            
            objects_packet = {
                "header": {"stamp": {"sec": header.stamp.sec, "nanosec": header.stamp.nanosec}},
                "camera_meta": self.camera_metadata["front_left"],
                "detections": []
            }

            for t_id, box in active_dynamic.items():
                x, y, w, h = box
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w_max, x + w), min(h_max, y + h)

                if (x2 - x1) <= 0 or (y2 - y1) <= 0: continue

                depth_patch = depth_np[y1:y2, x1:x2] if depth_np is not None else np.array([])
                median_depth = float(np.median(depth_patch)) if depth_patch.size > 0 else -1.0

                seg_patch = seg_np[y1:y2, x1:x2]
                if seg_patch.size > 0:
                    vals, counts = np.unique(seg_patch.flatten(), return_counts=True)
                    mode_seg = int(vals[np.argmax(counts)])
                else:
                    mode_seg = 0

                objects_packet["detections"].append({
                    "track_id": t_id,
                    "class_name": CLASSES.get(cls_id, "unknown").lower(),
                    "bbox": [x1, y1, x2, y2],
                    "median_depth": median_depth,
                    "semantic_id": mode_seg
                })

            if objects_packet["detections"] and not self.is_shutting_down:
                self.dynamic_objects_pub.publish(String(data=json.dumps(objects_packet)))

            light_packet = {"header": {"stamp": {"sec": header.stamp.sec, "nanosec": header.stamp.nanosec}}, "camera_meta": self.camera_metadata["front_left"], "detections": []}
            sign_payload = []

            for cls_id, box, conf in static_detections:
                x, y, w, h = box
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w_max, x + w), min(h_max, y + h)

                if (x2 - x1) < 4 or (y2 - y1) < 4: continue

                depth_patch = depth_np[y1:y2, x1:x2] if depth_np is not None else np.array([])
                median_depth = float(np.median(depth_patch)) if depth_patch.size > 0 else -1.0

                if cls_id == 2:
                    seg_patch = seg_np[y1:y2, x1:x2]
                    seg_confidence = float(np.sum(seg_patch == 7) / seg_patch.size) if seg_patch.size > 0 else 0.0

                    light_packet["detections"].append({
                        "bbox": [x1, y1, x2, y2],
                        "yolo_conf": float(conf),
                        "median_depth": median_depth,
                        "seg_verification_ratio": seg_confidence
                    })

                elif cls_id == 3:
                    if 'sign_classifier' in self.engines and raw_image_tensor is not None:
                        with torch.cuda.stream(self.engines['sign_classifier'].stream):
                            sign_crop = raw_image_tensor[:, :, y1:y2, x1:x2]
                            sign_resized = F.interpolate(sign_crop, size=(64, 64), mode='bilinear', align_corners=False)
                            sign_normalized = (sign_resized - 0.5) / 0.5

                            cnn_output = self.engines['sign_classifier'].enqueue_async(sign_normalized)
                            self.engines['sign_classifier'].stream.synchronize()

                            probabilities = F.softmax(cnn_output[0], dim=1)
                            conf_score, pred_idx = torch.max(probabilities, 1)
                            
                            pred_class = SIGN_CLASSES.get(int(pred_idx.item()), "unknown")
                            cnn_confidence = float(conf_score.item())
                    else:
                        pred_class, cnn_confidence = "UNCLASSIFIED", 0.0

                    sign_payload.append({
                        "class": pred_class,
                        "distance": median_depth,
                        "bbox": [x1, y1, x2, y2],
                        "x_center": float((x1 + x2) / 2.0),
                        "cnn_confidence": cnn_confidence
                    })

            if not self.is_shutting_down:
                if light_packet["detections"]: 
                    self.traffic_light_pub.publish(String(data=json.dumps(light_packet)))
                if sign_payload:
                    self.traffic_sign_pub.publish(String(data=json.dumps(sign_payload)))

        except Exception:
            if not self.is_shutting_down:
                self.get_logger().error(f"Detection Worker Thread Error:\n{traceback.format_exc()}")

    def _gpu_pipeline_worker(self, frames_to_process, wall_t0):
        if self.is_shutting_down: return
        try:
            t0 = time.perf_counter()
            gpu_tensors, baked_tensors, headers = {}, {}, {}
            fl_raw = None
            
            for cam, (header, bgr_array) in frames_to_process.items():
                pin_buf = self.pin_buffers[cam]
                norm_tensor, float_tensor, baked_tensor = self.preprocess_to_gpu(bgr_array, pin_buf)
                
                gpu_tensors[cam] = norm_tensor
                headers[cam] = header

                if cam in ['front_left', 'front_right']:
                    baked_tensors[cam] = baked_tensor
                
                if cam == 'front_left':
                    fl_raw = float_tensor
                    gpu_tensors['yolo'] = F.interpolate(fl_raw, size=(640, 640), mode='bilinear', align_corners=False)
                    
            self.t_prep.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            results = {}
            
            if 'front_left' in gpu_tensors:
                results['seg_front'] = self.engines['seg_front'].enqueue_async(gpu_tensors['front_left'])
                if 'yolo' in self.engines:
                    results['yolo'] = self.engines['yolo'].enqueue_async(gpu_tensors['yolo'])

            for cam in ['rear', 'side_left', 'side_right']:
                if cam in gpu_tensors:
                    results[f'{cam}_depth'] = self.engines[f'{cam}_depth'].enqueue_async(gpu_tensors[cam])
                    results[f'{cam}_seg']   = self.engines[f'{cam}_seg'].enqueue_async(gpu_tensors[cam])

            if 'front_left' in baked_tensors and 'front_right' in baked_tensors:
                results['stereonet'] = self.engines['stereonet'].enqueue_async(
                    baked_tensors['front_left'], 
                    baked_tensors['front_right']
                )

            synced = set()
            for name in results.keys():
                engine_key = name  
                if engine_key in self.engines and engine_key not in synced:
                    self.engines[engine_key].stream.synchronize()
                    synced.add(engine_key)
                    
            self.t_infer.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            fl_depth_tensor, fl_seg_tensor = None, None
            d_maps_gpu, s_maps_gpu = {}, {}

            if 'stereonet' in results:
                d_tensor = results['stereonet'][-1].squeeze()
                fl_depth_tensor = torch.clamp(320.0 / torch.clamp(d_tensor, 0.1, 250.0), 1.0, 250.0)
                d_maps_gpu['front_left'] = fl_depth_tensor
            
            if 'seg_front' in results:
                s_tensor = results['seg_front'][0].squeeze()
                fl_seg_tensor = torch.argmax(s_tensor, dim=0).to(torch.uint8)
                s_maps_gpu['front_left'] = fl_seg_tensor

            for cam in ['rear', 'side_left', 'side_right']:
                if f'{cam}_depth' in results:
                    d_tensor = results[f'{cam}_depth'][-1].squeeze()
                    d_maps_gpu[cam] = torch.clamp(d_tensor, 1.0, 250.0)
                if f'{cam}_seg' in results:
                    s_tensor = results[f'{cam}_seg'][0].squeeze()
                    s_maps_gpu[cam] = torch.argmax(s_tensor, dim=0).to(torch.uint8)

            bev_tensor = self.compute_bev_on_gpu(d_maps_gpu, s_maps_gpu)
            
            if not self.is_shutting_down:
                if 'front_left' in headers:
                    self.cpu_executor.submit(self.publish_bev_worker, bev_tensor, headers['front_left'])

                if 'front_left' in frames_to_process and (fl_depth_tensor is not None or fl_seg_tensor is not None):
                    self.cpu_executor.submit(self.process_and_publish_worker, 'front_left', fl_depth_tensor, fl_seg_tensor, headers['front_left'])
                
                # Check if we should publish other cameras too
                for cam in ['rear', 'side_left', 'side_right']:
                    if cam in frames_to_process:
                        d_t = d_maps_gpu.get(cam)
                        s_t = s_maps_gpu.get(cam)
                        if d_t is not None or s_t is not None:
                            self.cpu_executor.submit(self.process_and_publish_worker, cam, d_t, s_t, headers[cam])

                if 'yolo' in results:
                    yolo_tensor = results['yolo'][0].squeeze().T
                    if fl_seg_tensor is not None:
                        self.cpu_executor.submit(self.process_detections_worker, yolo_tensor, fl_depth_tensor, fl_seg_tensor, fl_raw, headers['front_left'])

                self.heartbeat_pub.publish(Bool(data=True))

            self.t_post.append((time.perf_counter() - t0) * 1000.0)
            self.t_wall.append((time.perf_counter() - wall_t0) * 1000.0)

            if 'front_left' in frames_to_process and self.frame_counts['front_left'] % 250 == 0:
                p_med    = np.median(self.t_prep)
                i_med    = np.median(self.t_infer)
                post_med = np.median(self.t_post)
                wall_med = np.median(self.t_wall)
                fps = 1000.0 / wall_med if wall_med > 0 else 0.0
                
                self.get_logger().info(
                    f"🏎️ PROFILER (Median) | Prep: {p_med:.1f}ms | Infer: {i_med:.1f}ms | Post(GPU BEV): {post_med:.1f}ms || "
                    f"Total: {wall_med:.1f}ms | Node FPS: {fps:.1f}"
                )
                self.t_prep, self.t_infer, self.t_post, self.t_wall = [], [], [], []

        except Exception:
            if not self.is_shutting_down:
                self.get_logger().error(f"Inference execution failed:\n{traceback.format_exc()}")
        finally:
            self.gpu_future = None  

    def inference_loop(self):
        if self.is_shutting_down or (self.gpu_future is not None and not self.gpu_future.done()):
            return
            
        frames_to_process = {}
        for cam in self.camera_names:
            if self.latest_frames[cam] is not None:
                frames_to_process[cam] = self.latest_frames[cam]
                self.latest_frames[cam] = None
                self.frame_counts[cam] += 1
                
        if not frames_to_process:
            return

        wall_t0 = time.perf_counter()
        try:
            self.gpu_future = self.cpu_executor.submit(self._gpu_pipeline_worker, frames_to_process, wall_t0)
        except RuntimeError:
            pass

    def destroy_node(self):
        self.is_shutting_down = True
        self.cpu_executor.shutdown(wait=False)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node) 
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()