#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Bool
from nav_msgs.msg import Odometry, Path
import json
import math

class VelocityProfileNode(Node):
    def __init__(self):
        super().__init__('velocity_profile_node')

        # Raised default speed cap from 60 km/h → 80 km/h
        self.max_allowed_speed = 60.0 / 3.6
        self.latest_targets = []
        
        self.ego_yaw_rate = 0.0
        self.ego_speed = 0.0  
        self.in_intersection = False
        self.intersection_hold_timer = 0  
        
        # State for debouncing intersection detection
        self.intersection_counter = 0
        
        # State for rate-limiting (slew rate) target speed
        self.last_target_speed = None
        
        self.fsm_speed_sub = self.create_subscription(Float32, '/behavior/target_cruise_speed', self.fsm_speed_callback, 10)
        self.ttc_sub = self.create_subscription(String, '/planning/fused_tracked_objects', self.objects_callback, 10)
        
        # Dynamic planning subscriptions
        self.odom_sub = self.create_subscription(Odometry, '/ekf/odometry', self.odom_callback, 10)
        self.local_path_sub = self.create_subscription(Path, '/planning/local_path', self.local_path_callback, 10)

        self.speed_pub = self.create_publisher(Float32, '/planning/target_speed', 10)
        self.hazard_pub = self.create_publisher(Bool, '/planning/hazard_detected', 10)

        self.timer = self.create_timer(0.05, self.publish_velocity_profile)

        self.get_logger().info("🏎️ VPG Constrained Optimizer Online (Dynamic ACC + Cornering Cap).")

    def fsm_speed_callback(self, msg: Float32):
        self.max_allowed_speed = msg.data

    def objects_callback(self, msg: String):
        try:
            self.latest_targets = json.loads(msg.data)
        except Exception:
            self.latest_targets = []

    def odom_callback(self, msg: Odometry):
        self.ego_yaw_rate = msg.twist.twist.angular.z
        self.ego_speed = abs(msg.twist.twist.linear.x)  

    def local_path_callback(self, msg: Path):
        # Look further ahead to catch upcoming maneuvers, not current steering wobble
        if len(msg.poses) < 30: 
            return
        
        # Check points 10 through 30 instead of the immediate 0-20
        max_y = max([abs(p.pose.position.y) for p in msg.poses[10:30]])
        
        # Debounce/hysteresis to prevent frame-to-frame flickering
        if max_y > 4.0:
            self.intersection_counter += 1
        else:
            self.intersection_counter -= 1
            
        # Clamp counter between 0 and 5
        self.intersection_counter = max(0, min(self.intersection_counter, 5))
        
        # Raised persistence requirement from 3 → 5 consecutive callbacks
        # to reduce false intersection detections triggering yield braking
        self.in_intersection = (self.intersection_counter >= 5)

    def publish_velocity_profile(self):
        hazard_detected = False
        target_speed = self.max_allowed_speed

        critical_ttc = 999.0
        critical_dist = 999.0
        crossing_vehicle_detected = False

        # 🎯 FIX 1: Locked to exactly 1.0m to completely ignore sidewalks/poles during turns
        lateral_limit = 1.0

        for obj in self.latest_targets:
            x, y, rel_spd, ttc, tid, cls_name = obj

            # INTERSECTION YIELD CHECK:
            if self.in_intersection and abs(y) > 3.5 and abs(y) < 12.0 and x < 12.0 and abs(rel_spd) > 2.5:
                time_to_cross = abs(y) / max(abs(rel_spd), 0.1)
                # Tightened yield window from 2.5s → 2.0s
                if time_to_cross < 2.0:
                    crossing_vehicle_detected = True

            # ACC LANE FILTER:
            # Cap lookahead distance at 60m (don't brake for cars in the next zip code)
            if 1.0 < x < 60.0 and abs(y) < lateral_limit:
                
                # Track closest absolute distance separately from TTC
                if x < critical_dist:
                    critical_dist = x

                # ONLY track TTC if the object is APPROACHING (ttc > 0)
                if 0.0 < ttc < critical_ttc:
                    critical_ttc = ttc

        # Stale-intersection timeout check
        # If we've been stopped at an intersection for >3 seconds, force-clear it
        if self.in_intersection and self.ego_speed < 0.5:
            self.intersection_hold_timer += 1
            if self.intersection_hold_timer > 60:  # 60 ticks × 0.05s = 3s
                self.in_intersection = False
                self.intersection_counter = 0
                self.intersection_hold_timer = 0
        else:
            self.intersection_hold_timer = 0

        # ---------------------------------------------------------
        # OPTIMIZATION SOLVER (Safety vs Comfort)
        # ---------------------------------------------------------

        # 🎯 FIX 2: Smooth cornering speed limit based on physical yaw rate.
        # This prevents the car from trying to turn at 30 km/h, which causes tire scrub and PID windup.
        turn_slowdown = min(abs(self.ego_yaw_rate) * 8.0, 5.0) # Max slowdown of 5 m/s
        safe_corner_speed = max(3.0, self.max_allowed_speed - turn_slowdown)

        # Apply the cornering cap to the raw target initially
        raw_target_speed = min(target_speed, safe_corner_speed)

        if self.in_intersection and crossing_vehicle_detected:
            hazard_detected = True
            raw_target_speed = 0.0

        # Emergency stop distance raised from 1.5m → 0.8m (actual bumper clearance).
        # Hard-stop TTC threshold lowered from 2.5s → 1.5s.
        elif critical_ttc < 1.5 or critical_dist < 0.8:
            hazard_detected = True
            raw_target_speed = 0.0

        elif critical_ttc < 6.0:
            # Soft ACC slowdown
            # Smoothing window now anchored at 1.5s (matching hard-stop threshold above)
            t = (critical_ttc - 1.5) / 4.5
            smooth_factor = t * t * (3.0 - 2.0 * t)
            # Make sure ACC calculates based on the corner-safe speed, not the absolute max speed
            acc_speed = safe_corner_speed * smooth_factor
            raw_target_speed = min(raw_target_speed, acc_speed)

        raw_target_speed = min(raw_target_speed, self.max_allowed_speed)

        # SLEW RATE LIMITER
        if self.last_target_speed is None:
            self.last_target_speed = raw_target_speed
            
        max_decel_step = 3.0 * 0.05  # 0.15 m/s per tick
        max_accel_step = 2.0 * 0.05  # 0.10 m/s per tick
        
        speed_diff = raw_target_speed - self.last_target_speed
        
        if speed_diff > max_accel_step:
            target_speed = self.last_target_speed + max_accel_step
        elif speed_diff < -max_decel_step:
            target_speed = self.last_target_speed - max_decel_step
        else:
            target_speed = raw_target_speed

        # Bypass slew limiter on hazard for emergency braking
        if hazard_detected:
            target_speed = 0.0

        self.last_target_speed = target_speed

        # Publish final speed
        spd_msg = Float32()
        spd_msg.data = float(target_speed)
        self.speed_pub.publish(spd_msg)

        haz_msg = Bool()
        haz_msg.data = hazard_detected
        self.hazard_pub.publish(haz_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VelocityProfileNode()
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