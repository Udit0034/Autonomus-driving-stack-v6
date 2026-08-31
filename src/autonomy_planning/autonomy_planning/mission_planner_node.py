#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Bool
import json
import csv
import carla
import sys
sys.path.append('/home/ubuntu/carla-0.9.16/PythonAPI/carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner

class MissionPlannerNode(Node):
    def __init__(self):
        super().__init__('mission_planner_node')
        
        self.cached_path_msg = None
        self.current_pose = None
        self.final_destination = None   # (x, y) in ROS frame
        self.last_completed_destination = None # 🎯 NEW: Anti-race condition tracker
        self.mission_complete_published = False

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(60.0)
        self.carla_map = self.client.get_world().get_map()
        
        self.route_planner = GlobalRoutePlanner(self.carla_map, sampling_resolution=0.5)

        self.path_log = open('/tmp/mission_planner_log.csv', 'w', newline='')
        self.path_log_writer = csv.writer(self.path_log)
        self.path_log_writer.writerow(['t_sec', 'wp_index', 'x', 'y'])

        self.path_pub = self.create_publisher(Path, '/planning/global_path', 10)
        self.status_pub = self.create_publisher(Bool, '/planning/mission_complete', 10)

        self.odom_sub = self.create_subscription(Odometry, '/ekf/odometry', self.odom_callback, 10)
        
        self.plan_sub = self.create_subscription(String, '/carla/hero/global_plan', self.global_plan_callback, 10)
        
        self.publish_timer = self.create_timer(0.1, self.publish_cached_path)
        
        self.get_logger().info("🗺️ Mission Planner Active. Listening for multi-waypoint plan from CARLA Node...")

        self.final_destination = None   # (x, y) in ROS frame
        self.mission_complete_published = False

    def odom_callback(self, msg: Odometry):
        self.current_pose = (msg.pose.pose.position.x, -msg.pose.pose.position.y)
        
        if (self.final_destination is not None and 
            not self.mission_complete_published and
            self.cached_path_msg is not None):
            
            dx = self.current_pose[0] - self.final_destination[0]
            dy = self.current_pose[1] - self.final_destination[1]
            
            if (dx*dx + dy*dy) < 25.0:  
                if not self.mission_complete_published:
                    done = Bool()
                    done.data = True
                    self.status_pub.publish(done)
                    self.mission_complete_published = True
                    self.get_logger().info("✅ Destination reached — Requesting next waypoint...")
                    
                    # 🎯 CRITICAL FIX: Save the destination we just hit before wiping it
                    self.last_completed_destination = self.final_destination 
                    
                    self.cached_path_msg = None
                    self.final_destination = None

    def global_plan_callback(self, msg: String):
        # 1. If we already built the path, ignore all incoming messages natively
        if self.cached_path_msg is not None:
            return
            
        try:
            data = json.loads(msg.data)
            
            # 2. Sanitize: If a ghost payload from the old version arrives, drop it
            if isinstance(data, list):
                self.get_logger().warn("Dropped outdated array payload. Waiting for dictionary format...")
                return
                
            spawn = data.get("spawn")
            destinations = data.get("destinations")
            
            if spawn is None or not destinations:
                self.get_logger().warn(f"Incomplete payload received. Raw data: {msg.data}")
                return
                
            # 🎯 NEW: Check if this is a stale broadcast of the route we JUST finished
            last = destinations[-1]
            incoming_dest = (float(last["x"]), float(last["y"]))
            
            if self.last_completed_destination is not None:
                dist_sq = (incoming_dest[0] - self.last_completed_destination[0])**2 + (incoming_dest[1] - self.last_completed_destination[1])**2
                if dist_sq < 1.0:  # If it's less than 1 meter away, it's the exact same target!
                    return         # Drop the stale message silently
                    
            self.execute_mission_plan(spawn, destinations)
            
        except Exception as e:
            self.get_logger().error(f"Failed to parse mission payload: {e} | Raw Data: {msg.data}")

    def execute_mission_plan(self, spawn, destinations):
        current_start = carla.Location(x=spawn[0], y=-spawn[1], z=0.0)
        all_waypoints = []
        
        self.get_logger().info(f"🔄 Plotting route from Spawn: ({spawn[0]:.2f}, {spawn[1]:.2f}) through {len(destinations)} waypoints...")
        
        for dest in destinations:
            raw_end = carla.Location(x=dest["x"], y=-dest["y"], z=0.0)
            
            start_wp = self.carla_map.get_waypoint(current_start, project_to_road=True, lane_type=carla.LaneType.Driving)
            end_wp   = self.carla_map.get_waypoint(raw_end, project_to_road=True, lane_type=carla.LaneType.Driving)

            if not start_wp or not end_wp:
                self.get_logger().error(f"❌ Could not snap to a valid driving lane for segment ending at ({dest['x']:.2f}, {dest['y']:.2f})")
                continue

            route = self.route_planner.trace_route(start_wp.transform.location, end_wp.transform.location)
            
            if route:
                for wp, _ in route:
                    all_waypoints.append((wp.transform.location.x, wp.transform.location.y))
                    
            current_start = raw_end
            
        if all_waypoints:
            self.build_global_path(all_waypoints)
            self.get_logger().info(f"✅ CARLA multi-segment route matched! {len(all_waypoints)} waypoints traced.")
        else:
            self.get_logger().error("❌ CARLA route planner returned no route.")
        # Store final destination for completion detection
        last = destinations[-1]
        self.final_destination = (float(last["x"]), float(last["y"]))

    def build_global_path(self, waypoints):
        path_msg = Path()
        path_msg.header.frame_id = "map" 

        for x, y in waypoints:
            pose = PoseStamped()
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y) 
            pose.pose.position.z = 0.0 
            pose.pose.orientation.w = 1.0 
            path_msg.poses.append(pose)

        self.cached_path_msg = path_msg

        # Log Path
        t = self.get_clock().now().nanoseconds / 1e9
        for i, pose in enumerate(path_msg.poses):
            self.path_log_writer.writerow([t, i, pose.pose.position.x, pose.pose.position.y])
        self.path_log.flush()
        # Tell the Behavior FSM that the new mission has started
        resume_msg = Bool()
        resume_msg.data = False
        self.status_pub.publish(resume_msg)
        self.mission_complete_published = False

    def publish_cached_path(self):
        if self.cached_path_msg is not None:
            current_time = self.get_clock().now().to_msg()
            self.cached_path_msg.header.stamp = current_time
            for pose in self.cached_path_msg.poses: pose.header.stamp = current_time
            self.path_pub.publish(self.cached_path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MissionPlannerNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()