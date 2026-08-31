#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>

class SemanticOccupancyGridNode : public rclcpp::Node {
public:
    SemanticOccupancyGridNode() : Node("semantic_occupancy_grid_node") {
        // Grid setup (40m x 40m)
        grid_res_ = 0.2f;
        grid_width_ = 200;
        grid_height_ = 200;
        grid_cx_ = grid_width_ / 2;
        grid_cy_ = grid_height_ / 2;

        // BEV Map Bounds (Matched to Python GPU Projection)
        bev_x_max_ = 50.0f;
        bev_x_min_ = -30.0f;
        bev_y_max_ = 25.0f;
        bev_y_min_ = -25.0f;

        // Subscribe directly to the unified BEV stream
        bev_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/dashboard/pred/bev", 10, std::bind(&SemanticOccupancyGridNode::bev_callback, this, std::placeholders::_1));

        map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/perception/semantic_costmap", 10);

        RCLCPP_INFO(this->get_logger(), "🚀 BEV-Optimized Semantic Occupancy Grid (Single-Channel Mono8) Online.");
    }

private:
    float grid_res_, bev_x_max_, bev_x_min_, bev_y_max_, bev_y_min_;
    int grid_width_, grid_height_, grid_cx_, grid_cy_;
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr bev_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;

    void bev_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv::Mat bev;

        if (msg->encoding == "mono8" || msg->encoding == "8UC1") {
            bev = cv_bridge::toCvCopy(msg, "mono8")->image;
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                "⚠️ Received 3-channel RGB BEV. Please update Python node to output 1-channel mono8 for maximum performance!");
            auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            cv::cvtColor(cv_ptr->image, bev, cv::COLOR_BGR2GRAY); // Desperate fallback, destroys labels
        }

        if (bev.empty()) return;

        std::vector<int8_t> grid_data(grid_width_ * grid_height_, -1);
        
        float x_range = bev_x_max_ - bev_x_min_;
        float y_range = bev_y_max_ - bev_y_min_;

        // O(N) Direct 2D mapping — completely bypasses 3D raycasting overhead
        for (int r = 0; r < bev.rows; ++r) {
            float x_metric = bev_x_max_ - (static_cast<float>(r) / bev.rows) * x_range;
            int gy = grid_cy_ - static_cast<int>(x_metric / grid_res_);
            
            if (gy < 0 || gy >= grid_height_) continue;

            // Direct memory pointer access for the row (blindingly fast)
            const uint8_t* row_ptr = bev.ptr<uint8_t>(r);

            for (int c = 0; c < bev.cols; ++c) {
                float y_metric = bev_y_min_ + (static_cast<float>(c) / bev.cols) * y_range;
                int gx = grid_cx_ + static_cast<int>(y_metric / grid_res_); 

                if (gx < 0 || gx >= grid_width_) continue;

                uint8_t s = row_ptr[c]; // Extract exact semantic ID
                int idx = gy * grid_width_ + gx;

                switch (s) {
    // --- LETHAL OBSTACLES (100) ---
    case 2:  case 3:  case 4:  case 5:  case 6:  case 7:  case 8:  case 9:  
    case 10: case 12: case 13: case 14: case 15: case 16: case 17: case 18: 
    case 19: case 20: case 21: case 22: case 23: case 26: case 27: case 28:
        grid_data[idx] = 100; 
        break;

    // --- SHARED / CAUTION AREAS (60) ---
    case 25: 
        if (grid_data[idx] != 100) {
            grid_data[idx] = 60; 
        }
        break;

    // --- DRIVABLE / FREE SPACE (0) ---
    case 1:  
    case 24: 
        if (grid_data[idx] != 100 && grid_data[idx] != 60) {
            grid_data[idx] = 0; 
        }
        break;

    // --- IGNORE (0: Unlabeled, 11: Sky) ---
    default:
        break; 
}
            }
        }

        // Stamp and Ship
        auto msg_out = nav_msgs::msg::OccupancyGrid();
        msg_out.header.stamp = this->get_clock()->now();
        msg_out.header.frame_id = "ego_vehicle";
        msg_out.info.resolution = grid_res_;
        msg_out.info.width = grid_width_;
        msg_out.info.height = grid_height_;
        msg_out.info.origin.position.x = -static_cast<float>(grid_cx_) * grid_res_;
        msg_out.info.origin.position.y = -static_cast<float>(grid_cy_) * grid_res_;
        msg_out.info.origin.orientation.w = 1.0;
        msg_out.data = grid_data;

        map_pub_->publish(msg_out);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SemanticOccupancyGridNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}