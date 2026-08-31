#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

class LidarOccupancyGridNode : public rclcpp::Node {
public:
    LidarOccupancyGridNode() : Node("lidar_occupancy_grid_node"), 
        grid_width_(240), grid_height_(240), grid_res_(0.25), grid_size_m_(60.0),
        slam_initialized_(false) {
        
        local_log_odds_.assign(grid_width_ * grid_height_, 0.0f);
        slam_pose_ = Eigen::Matrix4d::Identity();

        // 🎯 Upgrade 1: Subscribe directly to your high-fidelity SLAM node output
        slam_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/slam/odometry", 10, std::bind(&LidarOccupancyGridNode::slam_callback, this, std::placeholders::_1));

        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/carla/hero/lidar_top/point_cloud", 10, std::bind(&LidarOccupancyGridNode::lidar_callback, this, std::placeholders::_1));
        
        grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/slam/local_costmap", 10);

        RCLCPP_INFO(this->get_logger(), "⚡ SLAM-Driven Local Occupancy Grid Active.");
    }

private:
    std::vector<float> local_log_odds_;
    int grid_width_, grid_height_;
    double grid_res_, grid_size_m_;

    Eigen::Matrix4d slam_pose_;
    bool slam_initialized_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr slam_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;

    void slam_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        Eigen::Quaterniond q(
            msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
        
        slam_pose_ = Eigen::Matrix4d::Identity();
        slam_pose_.block<3,3>(0,0) = q.toRotationMatrix();
        slam_pose_(0,3) = msg->pose.pose.position.x;
        slam_pose_(1,3) = msg->pose.pose.position.y;
        slam_pose_(2,3) = msg->pose.pose.position.z;
        
        slam_initialized_ = true;
    }

    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (!slam_initialized_) return; // Wait until SLAM provides a global reference frame
        
        double ego_wx = slam_pose_(0,3);
        double ego_wy = slam_pose_(1,3);

        // Clear the temporal array completely per frame to prevent trailing/ghosting
        std::fill(local_log_odds_.begin(), local_log_odds_.end(), 0.0f);

        // Calculate bottom-left corner of our floating grid window in world coordinates
        double origin_wx = ego_wx - (grid_size_m_ / 2.0);
        double origin_wy = ego_wy - (grid_size_m_ / 2.0);

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
            float lx = *iter_x;
            float ly = *iter_y;
            float lz = *iter_z;

            // Strict Vertical Filter: captures immediate collision hazards (vehicles, barriers)
            if (lz < -1.3 || lz > 1.5) continue;
            
            float dist = std::sqrt(lx*lx + ly*ly);
            if (dist < 2.0 || dist > (grid_size_m_ / 2.0)) continue;

            // Transform point from local vehicle chassis frame into world frame coordinates
            Eigen::Vector4d pt_body(lx, ly, lz, 1.0);
            Eigen::Vector4d pt_world = slam_pose_ * pt_body;

            // 🎯 Upgrade 2 & 3: Standard ROS alignment logic to eliminate inversion issues
            // Grid cell coordinates are mapped directly from the computed bottom-left world origin
            int row = static_cast<int>((pt_world(0) - origin_wx) / grid_res_); // Row increases with X (Forward)
            int col = static_cast<int>((pt_world(1) - origin_wy) / grid_res_); // Column increases with Y (Left)

            if (col >= 0 && col < grid_width_ && row >= 0 && row < grid_height_) {
                int idx = row * grid_width_ + col;
                local_log_odds_[idx] += 1.5f;
                if (local_log_odds_[idx] > 5.0f) local_log_odds_[idx] = 5.0f;
            }
        }

        auto grid_msg = std::make_shared<nav_msgs::msg::OccupancyGrid>();
        grid_msg->header.stamp = msg->header.stamp;
        grid_msg->header.frame_id = "map"; 
        
        grid_msg->info.resolution = grid_res_;
        grid_msg->info.width = grid_width_;
        grid_msg->info.height = grid_height_;
        
        grid_msg->info.origin.position.x = origin_wx;
        grid_msg->info.origin.position.y = origin_wy;
        grid_msg->info.origin.position.z = slam_pose_(2,3); // Keep grid height at base link height
        grid_msg->info.origin.orientation.w = 1.0;

        grid_msg->data.resize(grid_width_ * grid_height_);
        for (int i = 0; i < grid_width_ * grid_height_; ++i) {
            float prob = 1.0f - (1.0f / (1.0f + std::exp(local_log_odds_[i])));
            int8_t val = static_cast<int8_t>(prob * 100);
            if (prob > 0.48 && prob < 0.52) val = -1; // Keep unobserved space marked as unknown (-1)
            grid_msg->data[i] = val;
        }

        grid_pub_->publish(*grid_msg);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarOccupancyGridNode>());
    rclcpp::shutdown();
    return 0;
}