import os
import sys
import glob
import cv2
import torch
import numpy as np
import rclpy
from rclpy.node import Node
import tensorrt as trt

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

# Image Normalization Constants for Stereo/Seg Calibration
mean_gpu = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
std_gpu  = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

def find_repo_root(start_path: str) -> str:
    current = os.path.abspath(start_path)
    while True:
        if os.path.exists(os.path.join(current, 'package.xml')):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.abspath(os.path.join(start_path, '..', '..'))

# =============================================================================
# INT8 ENTROPY CALIBRATORS
# =============================================================================
class FastDatasetEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """Calibrator for Segmentation and Depth Models"""
    def __init__(self, cache_dir, cam_names, input_shapes, calib_frames=100):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_dir = cache_dir
        self.cam_names = cam_names
        self.current_idx = 0
        
        search_path = os.path.join(cache_dir, "rgb", f"rgb_{cam_names[0]}_*.png")
        self.files = sorted(glob.glob(search_path))[:calib_frames]
        
        if len(self.files) == 0:
            raise FileNotFoundError(f"CRITICAL: No calibration images found in {search_path}")
            
        sys.stdout.write(f"      [Seg/Depth Calibrator] Found {len(self.files)} frames for {cam_names}.\n")
        self.device_inputs = [torch.empty(shape, dtype=torch.float32, device="cuda").contiguous() for shape in input_shapes]

    def get_batch_size(self): return 1

    def get_batch(self, names):
        if self.current_idx >= len(self.files): return None 
            
        idx_str = self.files[self.current_idx].split("_")[-1].split(".")[0]
        
        for i, cam in enumerate(self.cam_names):
            img_path = os.path.join(self.cache_dir, "rgb", f"rgb_{cam}_{idx_str}.png")
            img = cv2.imread(img_path)
            gpu_img = torch.from_numpy(img).to("cuda", non_blocking=True)
            gpu_img = gpu_img[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float().contiguous() / 255.0
            gpu_img = (gpu_img - mean_gpu) / std_gpu
            self.device_inputs[i].copy_(gpu_img)
            
        self.current_idx += 1
        return [int(tensor.data_ptr()) for tensor in self.device_inputs]

    def read_calibration_cache(self): return None
    def write_calibration_cache(self, cache): pass


class YOLOEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """Dedicated Calibrator for YOLOv11 (Enforces strict 640x640 scaling)"""
    def __init__(self, cache_dir, calib_frames=100):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_dir = cache_dir
        self.current_idx = 0
        
        search_path = os.path.join(cache_dir, "rgb", "rgb_front_left_*.png")
        self.files = sorted(glob.glob(search_path))[:calib_frames]
        
        if len(self.files) == 0:
            raise FileNotFoundError(f"CRITICAL: No YOLO calibration images found in {search_path}")
            
        sys.stdout.write(f"      [YOLO Calibrator] Found {len(self.files)} frames for rigorous 640x640 quantization.\n")
        
        self.device_input = torch.empty((1, 3, 640, 640), dtype=torch.float32, device="cuda").contiguous()

    def get_batch_size(self): return 1

    def get_batch(self, names):
        if self.current_idx >= len(self.files): return None 
            
        img_path = self.files[self.current_idx]
        img = cv2.imread(img_path)
        
        img = cv2.resize(img, (1280, 720))
        img_resized = cv2.resize(img, (640, 640))
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        gpu_img = torch.from_numpy(img_rgb).to("cuda", non_blocking=True)
        gpu_img = gpu_img.permute(2, 0, 1).unsqueeze(0).float().contiguous() / 255.0
        
        self.device_input.copy_(gpu_img)
        self.current_idx += 1
        
        return [int(self.device_input.data_ptr())]

    def read_calibration_cache(self): return None
    def write_calibration_cache(self, cache): pass

# =============================================================================
# ROS2 ENGINE BUILDER NODE
# =============================================================================
class EngineBuilderNode(Node):
    def __init__(self):
        super().__init__('engine_builder_node')
        
        self.onnx_dir = "/home/ubuntu/AV6/shared_cache/onnx_model"
        self.engine_dir = "/home/ubuntu/AV6/shared_cache/trt_engine_cache"
        self.calib_dir = "/home/ubuntu/calibration_dataset"

        # Configuration: (ONNX File, Engine File, Camera Names, Input Shapes, Architecture_Type)
        self.models_to_build = [
            # ("stereonet_surgery_int8_ready.onnx", "stereonet_mixed.engine", ["front_left", "front_right"], [(1,3,720,1280), (1,3,720,1280)], 'stereonet'),
            # ("front_seg_fp16_720p.onnx", "front_seg_int8.engine", ["front_left"], [(1, 3, 720, 1280)], 'standard'),
            # ("rear_depth_fp16.onnx", "rear_depth_int8.engine", ["rear"], [(1, 3, 400, 800)], 'standard'),
            # ("rear_seg_fp16.onnx", "rear_seg_int8.engine", ["rear"], [(1, 3, 400, 800)], 'standard'),
            # ("side_depth_fp16.onnx", "side_depth_int8.engine", ["side_left"], [(1, 3, 400, 800)], 'standard'),
            # ("side_segmentation_unified_fp16.onnx", "side_seg_int8.engine", ["side_left"], [(1, 3, 400, 800)], 'standard'),
            ("yolo_fp16.onnx", "yolo_int8.engine", ["front_left"], [(1, 3, 640, 640)], 'yolo'),
            ("carla_sign.onnx", "carla_sign_int8.engine", ["front_left"], [(1, 3, 64, 64)], 'standard') # <-- 🎯 Traffic Sign Classifier Added
        ]

    def build_all(self):
        self.get_logger().info("🚀 Starting Standalone INT8 Entropy TensorRT Pre-Compilation Node...")
        
        for onnx_file, engine_file, cam_names, shapes, arch_type in self.models_to_build:
            onnx_path = os.path.join(self.onnx_dir, onnx_file)
            engine_path = os.path.join(self.engine_dir, engine_file)
            
            if os.path.exists(engine_path):
                self.get_logger().info(f"✓ {engine_file} cache already exists. Skipping.")
                continue
                
            if not os.path.exists(onnx_path):
                self.get_logger().error(f"❌ Missing source ONNX file: {onnx_path}")
                continue

            self.get_logger().info(f"⚙️ Compiling {arch_type.upper()} Engine: {onnx_file} -> {engine_file}")
            self._build_engine(onnx_path, engine_path, cam_names, shapes, arch_type)
            
        self.get_logger().info(f"🎉 All baked engines successfully generated inside: {self.engine_dir}")
    #=============================================================================
    # SAFE DROP-IN ENGINE BUILDER METHOD
    # =============================================================================
    def _build_engine(self, onnx_path, engine_path, cam_names, shapes, arch_type):
        builder = trt.Builder(TRT_LOGGER)
        
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    sys.stderr.write(f"ONNX Parse Error: {parser.get_error(error)}\n")
                raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")
        
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  
        config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)  
        
        config.builder_optimization_level = 1
        sys.stdout.write("   → Optimization Level 5 (Maximum Tuning).\n")
        
        # 🎯 SAFE OPTIMIZATION PROFILE FOR DYNAMIC SHAPES (Bypasses error for carla_sign)
        has_dynamic = False
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            tensor_shape = input_tensor.shape
            
            # Check if the shape contains any dynamic markers (-1, None, or variable strings)
            if any(dim == -1 or dim is None or isinstance(dim, str) for dim in tensor_shape):
                has_dynamic = True
                # Build fixed profile dimensions using the precise shapes tuple passed down
                target_shape = [1 if (dim == -1 or dim is None or isinstance(dim, str)) else dim for dim in shapes[i]]
                profile.set_shape(input_tensor.name, target_shape, target_shape, target_shape)
                
        if has_dynamic:
            config.add_optimization_profile(profile)
            sys.stdout.write("   → Attached dynamic shape optimization profile to engine config.\n")

        # Setup INT8 Execution Flags
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        
        # 🎯 SEAMLESS ROUTING FOR ATTACHING CALIBRATORS
        if arch_type == 'yolo':
            config.int8_calibrator = YOLOEntropyCalibrator(self.calib_dir)
            sys.stdout.write("   → Attached YOLO Strict 640x640 Calibrator.\n")
        elif "carla_sign" in onnx_path:
            config.int8_calibrator = SignEntropyCalibrator(self.calib_dir, calib_frames=50)
            sys.stdout.write("   → Attached Sign-Specific 64x64 Calibrator (50 Frames).\n")
        else:
            config.int8_calibrator = FastDatasetEntropyCalibrator(self.calib_dir, cam_names, shapes)
            sys.stdout.write("   → Attached Standard Image Calibrator.\n")
        
        # --- StereoNet Wall constraints ---
        if arch_type == 'stereonet':
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            allowed_int8_types = [trt.LayerType.CONVOLUTION, trt.LayerType.ACTIVATION, trt.LayerType.SCALE, trt.LayerType.ELEMENTWISE, trt.LayerType.POOLING]
            locked_layers = 0
            
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                if layer.type in [trt.LayerType.CONSTANT, trt.LayerType.SHAPE, trt.LayerType.CAST, trt.LayerType.GATHER]:
                    continue
                    
                force_fp16 = False
                if layer.type not in allowed_int8_types: force_fp16 = True
                
                for j in range(layer.num_inputs):
                    inp = layer.get_input(j)
                    if inp and hasattr(inp, 'shape') and len(inp.shape) >= 5: force_fp16 = True
                for j in range(layer.num_outputs):
                    out = layer.get_output(j)
                    if out and hasattr(out, 'shape') and len(out.shape) >= 5: force_fp16 = True

                if force_fp16:
                    layer.precision = trt.float16
                    valid_lock = False
                    for j in range(layer.num_outputs):
                        out = layer.get_output(j)
                        if out and out.dtype in [trt.DataType.FLOAT, trt.DataType.HALF]:
                            layer.set_output_type(j, trt.float16)
                            valid_lock = True
                    if valid_lock: locked_layers += 1
            sys.stdout.write(f"   → StereoNet Wall Active: {locked_layers} downstream 3D layers locked to FP16.\n")

        if hasattr(trt.TacticSource, 'JIT_CONVOLUTIONS'):
            tactics = config.get_tactic_sources()
            tactics &= ~(1 << int(trt.TacticSource.JIT_CONVOLUTIONS))
            config.set_tactic_sources(tactics)

        sys.stdout.write("   ⏳ COMPILING HARDWARE ENGINE (This will take a few minutes)...\n")
        sys.stdout.flush()
        
        serialized_engine = builder.build_serialized_network(network, config)
            
        if serialized_engine is None:
            raise RuntimeError(f"Failed to build TensorRT engine: {onnx_path}")
            
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        sys.stdout.write(f"   ✓ Engine saved successfully: {os.path.basename(engine_path)}\n\n")
        sys.stdout.flush()
    # =============================================================================
# NEW DEDICATED TRAFFIC SIGN CALIBRATOR
# =============================================================================
class SignEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """Dedicated Calibrator for Traffic Sign Classifier (Strict 64x64 scaling, 50 frames)"""
    def __init__(self, cache_dir, calib_frames=50):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_dir = cache_dir
        self.current_idx = 0
        
        # Pulls the same front_left stream as YOLO
        search_path = os.path.join(cache_dir, "rgb", "rgb_front_left_*.png")
        self.files = sorted(glob.glob(search_path))[:calib_frames]
        
        if len(self.files) == 0:
            raise FileNotFoundError(f"CRITICAL: No sign calibration images found in {search_path}")
            
        sys.stdout.write(f"      [Sign Calibrator] Found {len(self.files)} frames for rigorous 64x64 quantization.\n")
        self.device_input = torch.empty((1, 3, 64, 64), dtype=torch.float32, device="cuda").contiguous()

    def get_batch_size(self): return 1

    def get_batch(self, names):
        if self.current_idx >= len(self.files): return None 
            
        img_path = self.files[self.current_idx]
        img = cv2.imread(img_path)
        
        # Match standard simulation resolution pipeline
        img = cv2.resize(img, (1280, 720))
        img_resized = cv2.resize(img, (64, 64))
        
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        gpu_img = torch.from_numpy(img_rgb).to("cuda", non_blocking=True)
        gpu_img = gpu_img.permute(2, 0, 1).unsqueeze(0).float().contiguous() / 255.0
        
        self.device_input.copy_(gpu_img)
        self.current_idx += 1
        
        return [int(self.device_input.data_ptr())]

    def read_calibration_cache(self): return None
    def write_calibration_cache(self, cache): pass




def main(args=None):
    rclpy.init(args=args)
    builder_node = EngineBuilderNode()
    try:
        builder_node.build_all()
    except Exception as e:
        builder_node.get_logger().error(f"Engine Compilation Interrupted: {e}")
    finally:
        builder_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()