#!/usr/bin/env python3
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from std_srvs.srv import Trigger
from std_msgs.msg import String 
from std_msgs.msg import Bool
from carla_msgs.msg import CarlaEgoVehicleControl 

import json
import carla
import numpy as np
import math
import random
import sys
import time

CAMS = {
    "front_left": {"fov": 90, "w": 1280, "h": 720, "x": 1.4, "y": -0.25, "z": 1.5, "pitch": -11.0, "yaw": 0.0, "roll": 0.0},
    "front_right": {"fov": 90, "w": 1280, "h": 720, "x": 1.4, "y": 0.25, "z": 1.5, "pitch": -11.0, "yaw": 0.0, "roll": 0.0},
    # "rear": {"fov": 100, "w": 800, "h": 400, "x": -2.0, "y": 0.0, "z": 1.6, "pitch": -26.0, "yaw": 180.0, "roll": 0.0},
    # "side_left": {"fov": 120, "w": 800, "h": 400, "x": 0.0, "y": -0.8, "z": 1.8, "pitch": -41.0, "yaw": -90.0, "roll": 0.0},
    # "side_right": {"fov": 120, "w": 800, "h": 400, "x": 0.0, "y": 0.8, "z": 1.8, "pitch": -41.0, "yaw": 90.0, "roll": 0.0},
}

LIDAR_CFG = {
    "x": 0.0, "y": 0.0, "z": 2.10,
    "channels": 28, "upper_fov": 15.0, "lower_fov": -25.0,
    "range": 150.0,
    "rotation_frequency": 20.0, 
    "points_per_second": 100000 
}

RADAR_CFG = {
    "front": {"x": 2.0, "y": 0.0, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "h_fov": 30.0, "v_fov": 15.0, "range": 150.0},
    "rear":  {"x": -2.0, "y": 0.0, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": 180.0, "h_fov": 30.0, "v_fov": 15.0, "range": 100.0},
    "fl":    {"x": 1.8, "y": -0.8, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": -60.0, "h_fov": 100.0, "v_fov": 30.0, "range": 20.0},
    "fr":    {"x": 1.8, "y": 0.8, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": 60.0, "h_fov": 100.0, "v_fov": 30.0, "range": 20.0},
    "rl":    {"x": -1.8, "y": -0.8, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": -125.0, "h_fov": 98.0, "v_fov": 46.0, "range": 20.0},
    "rr":    {"x": -1.8, "y": 0.8, "z": 0.5, "roll": 0.0, "pitch": 0.0, "yaw": 125.0, "h_fov": 100.0, "v_fov": 30.0, "range": 20.0}
}

class CarlaNodeNative(Node):
    def __init__(self):
        super().__init__('carla_node')
        self.declare_parameter('debug', False)
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        self.get_logger().info(f"CARLA Native ROS 2 Mode | Debug: {self.debug} | SYNC MODE: TRUE (20Hz)")
        self._last_dest_idx = None
        self.actor_list = []
        self.npc_vehicles = []
        self.npc_pedestrians = []
        self.npc_pedestrian_controllers = []

        self.running = True
        self.ego_vehicle = None
        
        self.autopilot_enabled = False 
        self.route_generated = False
        
        self.spawn_start_time = None
        self.tm_port = 8000
        self.tm = None

        self.gt_odom_pub = self.create_publisher(Odometry, '/carla/hero/odometry', 10)
        self.altimeter_pub = self.create_publisher(PointStamped, '/carla/altimeter', 10)
        self.wheel_odom_pub = self.create_publisher(Odometry, '/carla/hero/wheel_odom', 10)
        
        self.global_plan_pub = self.create_publisher(String, '/carla/hero/global_plan', 10)
        self.mission_payload_str = None 
        
        self.control_sub = self.create_subscription(
            CarlaEgoVehicleControl, 
            '/carla/hero/vehicle_control_cmd', 
            self.control_callback, 
            10
        )
        self.mission_sub = self.create_subscription(
            Bool, '/planning/mission_complete', self.mission_complete_callback, 10
        )
        
        self.connect_to_carla()
        
        self.sync_timer = self.create_timer(0.05, self.publish_custom_odometry)
        
        self.spawn_pt = None
        self.mission_destinations = [] 
        self.srv_get_mission = self.create_service(Trigger, '/carla/get_mission', self.get_mission_callback)

    def mission_complete_callback(self, msg: Bool):
        # If mission complete is True and we currently have a route
        if msg.data and self.route_generated:
            self.get_logger().info("🔄 Destination reached! Wiping old route and generating a new one...")
            self.route_generated = False
            self.mission_payload_str = None  # Stop broadcasting the old plan

    def connect_to_carla(self):
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()
        if self.world.get_map().name.split('/')[-1] != 'Town01':
            self.world = self.client.load_world('Town01')
            
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        for tl in self.world.get_actors().filter('traffic.traffic_light'): 
            tl.set_state(carla.TrafficLightState.Green)
            tl.set_green_time(3600.0)

        self.tm = self.client.get_trafficmanager(self.tm_port)
        self.tm.set_synchronous_mode(True)
        self.tm.set_global_distance_to_leading_vehicle(2.5)
        self.tm.set_hybrid_physics_mode(True)
        self.tm.set_hybrid_physics_radius(50.0)

        self.spawn_ego_and_sensors()
        self.spawn_npcs()  
        self.setup_spectator_camera()

    def control_callback(self, msg: CarlaEgoVehicleControl):
        if not self.ego_vehicle or not self.ego_vehicle.is_alive or self.autopilot_enabled:
            return
            
        control = carla.VehicleControl()
        control.throttle = float(np.clip(msg.throttle, 0.0, 1.0))
        control.steer = float(np.clip(msg.steer, -1.0, 1.0))
        control.brake = float(np.clip(msg.brake, 0.0, 1.0))
        control.hand_brake = bool(msg.hand_brake)
        control.reverse = bool(msg.reverse)
        
        self.ego_vehicle.apply_control(control)

    def get_mission_callback(self, request, response):
        mission_data = {
            "spawn": self.spawn_pt,
            "destinations": self.mission_destinations
        }
        response.success = True
        response.message = json.dumps(mission_data)
        return response

    def setup_spectator_camera(self):
        self.spectator_offset = carla.Vector3D(x=-15.0, y=0.0, z=8.0)
        self.spectator_rotation = carla.Rotation(pitch=-20, yaw=0, roll=0)

    @staticmethod
    def euler_to_quaternion(yaw, pitch, roll):
        yaw, pitch, roll = math.radians(yaw), math.radians(pitch), math.radians(roll)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        return np.array([
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            sy * cp * sr + cy * sp * cr,
            sy * cp * cr - cy * sp * sr,
        ])

    def spawn_ego_and_sensors(self):
        bp_lib = self.world.get_blueprint_library()
        ego_bp = bp_lib.find('vehicle.tesla.model3')
        
        ego_bp.set_attribute('role_name', 'hero')
        ego_bp.set_attribute('ros_name', 'hero')
        
        spawn_points = self.world.get_map().get_spawn_points()
        
        for spawn_point in spawn_points[0:10]:
            try:
                self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
                self.spawn_start_time = time.time()
                self.spawn_pt = (spawn_point.location.x, -spawn_point.location.y)
                break
            except RuntimeError:
                continue
                
        self.actor_list.append(self.ego_vehicle)
        
        self.get_logger().info(
            f"✅ Hero vehicle spawned at X: {spawn_point.location.x:.2f}, "
            f"Y: {spawn_point.location.y:.2f}, Z: {spawn_point.location.z:.2f} | "
            f"Yaw: {spawn_point.rotation.yaw:.2f}°"
        )

        for name, ext in CAMS.items():
            tf = carla.Transform(carla.Location(x=ext['x'], y=ext['y'], z=ext['z']), carla.Rotation(pitch=ext['pitch'], yaw=ext['yaw'], roll=ext['roll']))
            cam_bp = bp_lib.find('sensor.camera.rgb')
            cam_bp.set_attribute('image_size_x', str(ext['w']))
            cam_bp.set_attribute('image_size_y', str(ext['h']))
            cam_bp.set_attribute('fov', str(ext['fov']))
            cam_bp.set_attribute('sensor_tick', '0.05')
            cam_bp.set_attribute('role_name', name)
            cam_bp.set_attribute('ros_name', name)
            
            cam = self.world.spawn_actor(cam_bp, tf, attach_to=self.ego_vehicle)
            cam.enable_for_ros() 
            self.actor_list.append(cam)
            
            if self.debug:
                if name == 'front_right': continue
                depth_bp = bp_lib.find('sensor.camera.depth')
                depth_bp.set_attribute('image_size_x', str(ext['w']))
                depth_bp.set_attribute('image_size_y', str(ext['h']))
                depth_bp.set_attribute('fov', str(ext['fov']))
                depth_bp.set_attribute('sensor_tick', '0.05')
                depth_bp.set_attribute('role_name', f'depth_{name}')
                depth_bp.set_attribute('ros_name', f'depth_{name}')
                depth = self.world.spawn_actor(depth_bp, tf, attach_to=self.ego_vehicle)
                depth.enable_for_ros()
                self.actor_list.append(depth)

                seg_bp = bp_lib.find('sensor.camera.semantic_segmentation')
                seg_bp.set_attribute('image_size_x', str(ext['w']))
                seg_bp.set_attribute('image_size_y', str(ext['h']))
                seg_bp.set_attribute('fov', str(ext['fov']))
                seg_bp.set_attribute('sensor_tick', '0.05')
                seg_bp.set_attribute('role_name', f'seg_{name}')
                seg_bp.set_attribute('ros_name', f'seg_{name}')
                seg = self.world.spawn_actor(seg_bp, tf, attach_to=self.ego_vehicle)
                seg.enable_for_ros()
                self.actor_list.append(seg)
            
        self.get_logger().info(f"✅ Cameras spawned.")

        imu_bp = bp_lib.find('sensor.other.imu')
        imu_bp.set_attribute('role_name', 'imu')
        imu_bp.set_attribute('ros_name', 'imu')
        imu_bp.set_attribute('sensor_tick', '0.05')
        imu = self.world.spawn_actor(imu_bp, carla.Transform(), attach_to=self.ego_vehicle)
        imu.enable_for_ros()
        self.actor_list.append(imu)

        gnss_front_bp = bp_lib.find('sensor.other.gnss')
        gnss_front_bp.set_attribute('role_name', 'gnss_front')
        gnss_front_bp.set_attribute('ros_name', 'gnss_front')
        gnss_front_bp.set_attribute('sensor_tick', '0.05')
        gnss_front = self.world.spawn_actor(gnss_front_bp, carla.Transform(carla.Location(x=1.0, z=1.5)), attach_to=self.ego_vehicle)
        gnss_front.enable_for_ros()
        self.actor_list.append(gnss_front)

        gnss_rear_bp = bp_lib.find('sensor.other.gnss')
        gnss_rear_bp.set_attribute('role_name', 'gnss_rear')
        gnss_rear_bp.set_attribute('ros_name', 'gnss_rear')
        gnss_rear_bp.set_attribute('sensor_tick', '0.05')
        gnss_rear = self.world.spawn_actor(gnss_rear_bp, carla.Transform(carla.Location(x=-1.0, z=1.5)), attach_to=self.ego_vehicle)
        gnss_rear.enable_for_ros()
        self.actor_list.append(gnss_rear)

        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', str(LIDAR_CFG['channels']))
        lidar_bp.set_attribute('points_per_second', str(LIDAR_CFG['points_per_second']))
        lidar_bp.set_attribute('rotation_frequency', str(LIDAR_CFG['rotation_frequency']))
        lidar_bp.set_attribute('range', str(LIDAR_CFG['range']))
        lidar_bp.set_attribute('upper_fov', str(LIDAR_CFG['upper_fov']))
        lidar_bp.set_attribute('lower_fov', str(LIDAR_CFG['lower_fov']))
        lidar_bp.set_attribute('sensor_tick', '0.05')
        lidar_bp.set_attribute('role_name', 'lidar_top')
        lidar_bp.set_attribute('ros_name', 'lidar_top')

        lidar_tf = carla.Transform(carla.Location(x=LIDAR_CFG['x'], y=LIDAR_CFG['y'], z=LIDAR_CFG['z']))
        lidar = self.world.spawn_actor(lidar_bp, lidar_tf, attach_to=self.ego_vehicle)
        lidar.enable_for_ros()
        self.actor_list.append(lidar)

    def spawn_npcs(self):
        bp_lib = self.world.get_blueprint_library()
        
        vehicle_bps = bp_lib.filter('vehicle.*')
        vehicle_bps = [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]
        
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        for i in range(min(15, len(spawn_points))):
            bp = random.choice(vehicle_bps)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            bp.set_attribute('role_name', 'autopilot')
            
            npc = self.world.try_spawn_actor(bp, spawn_points[i])
            if npc is not None:
                npc.set_autopilot(True, self.tm_port)
                self.npc_vehicles.append(npc)
                
        self.get_logger().info(f"✅ Spawned {len(self.npc_vehicles)} NPC vehicles on Autopilot.")

        walker_bps = bp_lib.filter('walker.pedestrian.*')
        controller_bp = bp_lib.find('controller.ai.walker')
        
        spawned_walkers = 0
        max_attempts = 100 
        attempts = 0
        
        while spawned_walkers < 0 and attempts < max_attempts:
            attempts += 1
            spawn_loc = self.world.get_random_location_from_navigation()
            if spawn_loc is None: continue
            
            spawn_transform = carla.Transform(spawn_loc)
            walker_bp = random.choice(walker_bps)
            
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
                
            walker = self.world.try_spawn_actor(walker_bp, spawn_transform)
            if walker is not None:
                controller = self.world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                if controller is not None:
                    self.npc_pedestrians.append(walker)
                    self.npc_pedestrian_controllers.append(controller)
                    spawned_walkers += 1
                else:
                    walker.destroy() 
                    
        self.world.tick()
        
        for controller in self.npc_pedestrian_controllers:
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(1.2 + random.random()) 
            
        self.get_logger().info(f"✅ Spawned {len(self.npc_pedestrians)} Pedestrians with AI walking active.")

    def publish_custom_odometry(self):
        if not self.running or self.ego_vehicle is None or not self.ego_vehicle.is_alive:
            return
        try:
            self.world.tick()

            # 1. GENERATE ONCE
            if not self.route_generated:
                if time.time() - self.spawn_start_time >= 5.0:
                    carla_map = self.world.get_map()
                    all_spawn_points = carla_map.get_spawn_points()
                    
                    available = [i for i, pt in enumerate(all_spawn_points)if self._last_dest_idx is None or i != self._last_dest_idx]
                    chosen_idx = random.choice(available)
                    self._last_dest_idx = chosen_idx
                    sampled_points = [all_spawn_points[chosen_idx]]
                    
                    # Fetch the live location to guarantee 'spawn' is NEVER null
                    live_loc = self.ego_vehicle.get_transform().location
                    self.spawn_pt = (live_loc.x, -live_loc.y)
                    
                    self.mission_destinations = [{"x": pt.location.x, "y": -pt.location.y} for pt in sampled_points]
                    
                    mission_payload = {
                        "spawn": self.spawn_pt,
                        "destinations": self.mission_destinations
                    }
                    self.mission_payload_str = json.dumps(mission_payload)
                    
                    if self.autopilot_enabled:
                        tm_waypoints_path = [pt.location for pt in sampled_points]
                        self.tm.set_path(self.ego_vehicle, tm_waypoints_path)
                        self.tm.vehicle_percentage_speed_difference(self.ego_vehicle, -20.0)
                        self.ego_vehicle.set_autopilot(True, self.tm_port)
                        self.get_logger().info(f"✅ Single Route Generated. CARLA Autopilot Engaged!")
                    else:
                        self.get_logger().info(f"✅ Single Target Broadcasted to ROS. C++ Controller Engaged!")
                        
                    self.route_generated = True

            # 2. BROADCAST CONTINUOUSLY (Robust Delivery)
            if self.route_generated and self.mission_payload_str is not None:
                plan_msg = String()
                plan_msg.data = self.mission_payload_str
                self.global_plan_pub.publish(plan_msg)

            ego_transform = self.ego_vehicle.get_transform()
            forward, right, up = ego_transform.get_forward_vector(), ego_transform.get_right_vector(), ego_transform.get_up_vector()
            
            offset = self.spectator_offset
            spectator_loc = carla.Location(
                x=ego_transform.location.x + forward.x * offset.x + right.x * offset.y + up.x * offset.z,
                y=ego_transform.location.y + forward.y * offset.x + right.y * offset.y + up.y * offset.z,
                z=ego_transform.location.z + forward.z * offset.x + right.z * offset.y + up.z * offset.z,
            )
            self.world.get_spectator().set_transform(carla.Transform(spectator_loc, carla.Rotation(pitch=self.spectator_rotation.pitch, yaw=ego_transform.rotation.yaw, roll=self.spectator_rotation.roll)))

            stamp = self.get_clock().now().to_msg()
            velocity = self.ego_vehicle.get_velocity()
            
            odom_msg = Odometry()
            odom_msg.header.stamp = stamp
            odom_msg.header.frame_id = 'odom'
            odom_msg.child_frame_id = 'base_link'
            odom_msg.pose.pose.position.x = ego_transform.location.x
            odom_msg.pose.pose.position.y = ego_transform.location.y
            odom_msg.pose.pose.position.z = ego_transform.location.z
            q = self.euler_to_quaternion(-ego_transform.rotation.yaw, -ego_transform.rotation.pitch, ego_transform.rotation.roll)
            odom_msg.pose.pose.orientation.w = float(q[0])
            odom_msg.pose.pose.orientation.x = float(q[1])
            odom_msg.pose.pose.orientation.y = float(q[2])
            odom_msg.pose.pose.orientation.z = float(q[3])
            odom_msg.twist.twist.linear.x = velocity.x
            odom_msg.twist.twist.linear.y = -velocity.y
            odom_msg.twist.twist.linear.z = velocity.z
            self.gt_odom_pub.publish(odom_msg)

            gt_z_noisy = ego_transform.location.z + random.gauss(0.0, 0.3)
            alt_msg = PointStamped()
            alt_msg.header.stamp = stamp
            alt_msg.header.frame_id = 'odom'
            alt_msg.point.z = gt_z_noisy
            self.altimeter_pub.publish(alt_msg)

            gt_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            odom_speed = gt_speed * 1.02 + random.gauss(0.0, 0.1) 
            wheel_msg = Odometry()
            wheel_msg.header.stamp = stamp
            wheel_msg.header.frame_id = 'base_link'
            wheel_msg.twist.twist.linear.x = odom_speed
            self.wheel_odom_pub.publish(wheel_msg)
            
        except Exception as e:
            self.get_logger().error(f"publish_custom_odometry failed: {e}", throttle_duration_sec=2.0)

    def destroy_node(self):
        self.running = False
        self.get_logger().info("🛑 Shutting down. Cleaning up all actors...")
        
        try:
            self.ego_vehicle.set_autopilot(False, self.tm_port)
        except Exception: pass
        try:
            self.sync_timer.cancel()
        except Exception: pass
        
        for controller in self.npc_pedestrian_controllers:
            try:
                if controller.is_alive:
                    controller.stop()
                    controller.destroy()
            except Exception: pass
        self.npc_pedestrian_controllers.clear()

        for walker in self.npc_pedestrians:
            try:
                if walker.is_alive: walker.destroy()
            except Exception: pass
        self.npc_pedestrians.clear()

        for npc_veh in self.npc_vehicles:
            try:
                if npc_veh.is_alive:
                    npc_veh.set_autopilot(False, self.tm_port)
                    npc_veh.destroy()
            except Exception: pass
        self.npc_vehicles.clear()

        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = 0.0
            self.world.apply_settings(settings)
        except Exception: pass
        
        for actor in list(self.actor_list):
            try:
                if actor.is_alive: actor.destroy()
            except Exception: pass
        self.actor_list.clear()
        
        self.get_logger().info("✅ Cleanup complete.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CarlaNodeNative()
    executor = MultiThreadedExecutor()
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