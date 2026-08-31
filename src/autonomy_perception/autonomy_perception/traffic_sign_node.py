#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
import json

class TrafficSignNode(Node):
    def __init__(self):
        super().__init__('traffic_sign_node')

        # Fallback Default: 30 km/h converted to m/s (~8.33 m/s)
        self.default_speed_ms = 30.0 / 3.6
        self.current_speed_limit_ms = self.default_speed_ms
        
        # Distance threshold to consider a sign active (meters)
        self.max_consideration_distance = 25.0 

        # Subscribes to the JSON output of the modified inference node
        self.sign_sub = self.create_subscription(
            String, 
            '/perception/sign_detections', 
            self.sign_callback, 
            10
        )

        # Publishes regulated state constraint to the state executive / planner
        self.limit_pub = self.create_publisher(Float32, '/planning/speed_limit_constraint', 10)
        
        # 10 Hz Execution loop
        self.timer = self.create_timer(0.1, self.publish_speed_constraint)
        self.get_logger().info("🛑 Traffic Sign State Processing Node Active (Default: 30 km/h).")

    def sign_callback(self, msg: String):
        try:
            # Expected payload format from updated inference crop engine:
            # [{"class": "speed_60", "distance": 12.4, "x_center": 0.2}, ...]
            detections = json.loads(msg.data)
        except Exception:
            return

        if not detections:
            return

        closest_sign = None
        min_distance = float('inf')

        # 1. Spatial Resolution: Parse through all visible signs to find the closest one
        for det in detections:
            dist = det.get('distance', 999.0)
            if dist < min_distance and dist < self.max_consideration_distance:
                min_distance = dist
                closest_sign = det

        # 2. State Machine Update
        if closest_sign:
            sign_type = closest_sign.get('class', 'unknown')
            self.get_logger().info(f"🎯 Closest Traffic Sign Identified: {sign_type} at {min_distance:.2f}m")
            
            # Map detected class to velocity values (m/s)
            if sign_type == "speed_30":
                self.current_speed_limit_ms = 30.0 / 3.6
            elif sign_type == "speed_40":
                self.current_speed_limit_ms = 40.0 / 3.6
            elif sign_type == "speed_60":
                self.current_speed_limit_ms = 60.0 / 3.6
            elif sign_type == "speed_90":
                self.current_speed_limit_ms = 90.0 / 3.6
            elif sign_type == "stop":
                # Let the executive or VPG handle the stop line smoothly; target safe creep
                self.current_speed_limit_ms = 0.0
            # Note: If unknown, we keep our last active state constraint persistent

    def publish_speed_constraint(self):
        msg = Float32()
        msg.data = float(self.current_speed_limit_ms)
        self.limit_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrafficSignNode()
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