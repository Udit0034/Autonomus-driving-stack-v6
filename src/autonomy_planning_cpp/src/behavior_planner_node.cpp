#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp> 
#include <cmath>
#include <mutex>
#include <algorithm>
#include <vector>
#include <utility>

class BehaviorPlannerNode : public rclcpp::Node {
public:
    BehaviorPlannerNode() : Node("behavior_planner_node") {
        map_speed_limit_ms_ = 30.0f / 3.6f;
        ideal_lane_offset_ = 0.0f; 
        lane_offset_valid_ = false; 
        
        ego_x_ = 0.0f; ego_y_ = 0.0f; 

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ekf/odometry", 10, std::bind(&BehaviorPlannerNode::odom_callback, this, std::placeholders::_1));
        sign_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "/planning/speed_limit_constraint", 10, std::bind(&BehaviorPlannerNode::sign_callback, this, std::placeholders::_1));
        lane_offset_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "/perception/lane_center_offset", 10, std::bind(&BehaviorPlannerNode::lane_offset_callback, this, std::placeholders::_1));
        lane_valid_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/perception/lane_center_offset_valid", 10, std::bind(&BehaviorPlannerNode::lane_valid_callback, this, std::placeholders::_1));
        global_path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/planning/global_path", 10, std::bind(&BehaviorPlannerNode::global_path_callback, this, std::placeholders::_1));

        behavior_speed_pub_ = this->create_publisher<std_msgs::msg::Float32>("/behavior/target_cruise_speed", 10);
        behavior_lane_pub_ = this->create_publisher<std_msgs::msg::Float32>("/behavior/target_lane_offset", 10);

        timer_ = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&BehaviorPlannerNode::evaluate_fsm, this));
        RCLCPP_INFO(this->get_logger(), "🧠 Barebones Behavior Planner (Locked Cruise + Strict Turn Offsets) Online.");
    }

private:
    std::mutex state_mutex_;
    float map_speed_limit_ms_, ideal_lane_offset_;
    bool lane_offset_valid_;
    
    float ego_x_, ego_y_;
    std::vector<std::pair<float,float>> global_path_pts_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sign_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr lane_offset_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr lane_valid_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr global_path_sub_; 
    
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr behavior_speed_pub_, behavior_lane_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        ego_x_ = msg->pose.pose.position.x;
        ego_y_ = msg->pose.pose.position.y;
    }

    void sign_callback(const std_msgs::msg::Float32::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(state_mutex_); map_speed_limit_ms_ = msg->data;
    }

    void lane_offset_callback(const std_msgs::msg::Float32::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(state_mutex_); 
        float raw_offset = std::clamp(msg->data, -1.0f, 1.0f);
        constexpr float alpha = 0.05f;
        if (!lane_offset_valid_) ideal_lane_offset_ = raw_offset;
        else ideal_lane_offset_ = (alpha * raw_offset) + ((1.0f - alpha) * ideal_lane_offset_);
    }

    void lane_valid_callback(const std_msgs::msg::Bool::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(state_mutex_); lane_offset_valid_ = msg->data;
    }

    void global_path_callback(const nav_msgs::msg::Path::SharedPtr msg) {
        std::vector<std::pair<float,float>> pts;
        pts.reserve(msg->poses.size());
        for (const auto& p : msg->poses) pts.emplace_back(p.pose.position.x, p.pose.position.y);
        std::lock_guard<std::mutex> lock(state_mutex_);
        global_path_pts_ = std::move(pts);
    }

    void evaluate_fsm() {
        float speed_limit, lane_offset;
        bool lane_valid; 
        float ego_x, ego_y;
        std::vector<std::pair<float,float>> g_pts;

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            speed_limit = map_speed_limit_ms_;
            lane_offset = ideal_lane_offset_; lane_valid = lane_offset_valid_; 
            ego_x = ego_x_; ego_y = ego_y_;
            g_pts = global_path_pts_; 
        }

        // 1. Calculate Curvature cleanly
        float global_fwd_curv = 0.0f;
        if (g_pts.size() >= 3) {
            size_t nearest = 0;
            float min_d = std::numeric_limits<float>::max();
            for (size_t i = 0; i < g_pts.size(); ++i) {
                float d = std::hypot(g_pts[i].first - ego_x, g_pts[i].second - ego_y);
                if (d < min_d) { min_d = d; nearest = i; }
            }
            
            float acc = 0.0f;
            for (size_t i = nearest; i + 2 < g_pts.size(); ++i) {
                acc += std::hypot(g_pts[i+1].first - g_pts[i].first, g_pts[i+1].second - g_pts[i].second);
                if (acc < 5.0f) continue;  
                if (acc > 25.0f) break;    
                
                float x0=g_pts[i].first,   y0=g_pts[i].second;
                float x1=g_pts[i+1].first, y1=g_pts[i+1].second;
                float x2=g_pts[i+2].first, y2=g_pts[i+2].second;
                float a=std::hypot(x1-x0,y1-y0), b=std::hypot(x2-x1,y2-y1);
                
                if (a<1e-3f || b<1e-3f) continue;
                float area2 = std::abs((x1-x0)*(y2-y0)-(x2-x0)*(y1-y0));
                float c = std::hypot(x2-x0,y2-y0);
                if (c < 1e-3f) continue;
                
                global_fwd_curv = std::max(global_fwd_curv, (2.0f*area2)/(a*b*c));
            }
        }

        // 2. Strict Turn Offset Disable (Fixes the wide swing seen in image_f14ebc.png)
        float lane_weight = 1.0f;
        if (global_fwd_curv > 0.08f) { 
            // Instantly disable semantic lane following if a turn is detected
            lane_weight = 0.0f;
        }

        float deadbanded_offset = (std::abs(lane_offset) > 0.15f) ? lane_offset : 0.0f;
        float target_lane = lane_valid ? deadbanded_offset * lane_weight : 0.0f;

        // 3. Absolute Constant Speed (Fixes the hiccups seen in image_f152fc.png)
        // No traffic lights, no sharp turn caps. It feeds directly into the VPG node.
        float target_speed = speed_limit; 

        std_msgs::msg::Float32 speed_msg, lane_msg;
        speed_msg.data = target_speed; lane_msg.data = target_lane;
        behavior_speed_pub_->publish(speed_msg); behavior_lane_pub_->publish(lane_msg);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BehaviorPlannerNode>();
    rclcpp::spin(node); rclcpp::shutdown(); return 0;
}