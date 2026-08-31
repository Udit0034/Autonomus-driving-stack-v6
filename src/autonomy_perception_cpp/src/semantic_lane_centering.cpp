#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>

using std::placeholders::_1;

class SemanticLaneCentering : public rclcpp::Node {
public:
    SemanticLaneCentering() : Node("semantic_lane_centering") {
        filtered_offset_ = 0.0f; 
        alpha_ = 0.3f; 
        invalid_count_ = 0;

        offset_pub_ = this->create_publisher<std_msgs::msg::Float32>("/perception/lane_center_offset", 10);
        valid_pub_ = this->create_publisher<std_msgs::msg::Bool>("/perception/lane_center_offset_valid", 10);

        bev_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/dashboard/pred/bev", 10, std::bind(&SemanticLaneCentering::bev_callback, this, _1));

        RCLCPP_INFO(this->get_logger(), "🛣️ Semantic Lane Centering Node (RHD Sign-Corrected) Online.");
    }

private:
    float filtered_offset_, alpha_;
    int invalid_count_; 

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr bev_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr offset_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr valid_pub_;

    void publish_offset(bool valid_frame) {
        if (!valid_frame) {
            invalid_count_++;
            if (invalid_count_ > 40) filtered_offset_ = 0.0f;
        } else {
            invalid_count_ = 0;
        }
        
        auto msg = std_msgs::msg::Float32(); 
        msg.data = filtered_offset_;
        offset_pub_->publish(msg);
        
        std_msgs::msg::Bool valid_msg; 
        valid_msg.data = valid_frame;
        valid_pub_->publish(valid_msg);
    }

    void bev_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        try {
            cv::Mat bev;
            if (msg->encoding == "mono8" || msg->encoding == "8UC1") {
                bev = cv_bridge::toCvCopy(msg, "mono8")->image;
            } else {
                return;
            }

            if (bev.empty()) {
                publish_offset(false);
                return;
            }

            float pixels_per_meter_x = bev.rows / 80.0f;
            float pixels_per_meter_y = bev.cols / 50.0f;

            // Scan band 2m–10m ahead
            int row_start = static_cast<int>((50.0f - 10.0f) * pixels_per_meter_x);
            int row_end   = static_cast<int>((50.0f - 2.0f)  * pixels_per_meter_x);
            row_start = std::clamp(row_start, 0, bev.rows - 1);
            row_end   = std::clamp(row_end,   0, bev.rows - 1);

            std::vector<int> lines_hist(bev.cols, 0);
            std::vector<int> road_hist(bev.cols, 0);
            int total_line_pixels = 0;

            for (int r = row_start; r <= row_end; ++r) {
                const uint8_t* row_ptr = bev.ptr<uint8_t>(r);
                for (int c = 0; c < bev.cols; ++c) {
                    if (row_ptr[c] == 24) {
                        lines_hist[c]++;
                        total_line_pixels++;
                    } else if (row_ptr[c] == 1) {
                        road_hist[c]++;
                    }
                }
            }

            if (total_line_pixels < 8) {
                publish_offset(false);
                return;
            }

            // CORRECTED COORDINATE SYSTEM: 
            // Negative sign ensures Cols 0-249 (Left) are +Y, Cols 251-500 (Right) are -Y
            auto col_to_y = [&](int col) {
                return -((static_cast<float>(col) / pixels_per_meter_y) - 25.0f);
            };

            int center_col = static_cast<int>(25.0f * pixels_per_meter_y); // ~250

            // Step 1: Find LEFT boundary (Center Divider Line)
            // Scan from vehicle center outward to the left (250 down to 0)
            int left_col = -1;
            for (int c = center_col; c >= 0; --c) {
                float y = col_to_y(c);
                if (y < 0.2f || y > 3.5f) continue; // Expect left line between +0.2m and +3.5m
                if (lines_hist[c] >= 3) {
                    left_col = c;
                    break; // Right-most edge of the left line
                }
            }

            // Step 2: Find RIGHT boundary (Road Edge)
            // Scan from far right inward toward vehicle (499 down to 250)
            int right_col = -1;
            for (int c = bev.cols - 1; c >= center_col; --c) {
                float y = col_to_y(c);
                if (y > -0.2f || y < -4.5f) continue; // Expect right road edge between -0.2m and -4.5m
                if (road_hist[c] >= 5) { // Slightly higher threshold to ignore stray pixels
                    right_col = c;
                    break; // Right-most legitimate road pixel
                }
            }

            // Step 3: Convert to metric Y positions
            float left_bound  = (left_col >= 0) ? col_to_y(left_col) : std::numeric_limits<float>::quiet_NaN();
            float right_bound = (right_col >= 0) ? col_to_y(right_col) : std::numeric_limits<float>::quiet_NaN();

            // Debug logging removed

            // Step 4: Fallbacks for right-hand drive
            bool has_left  = !std::isnan(left_bound);
            bool has_right = !std::isnan(right_bound);

            if (!has_left && !has_right) { publish_offset(false); return; }
            
            // If one is missing, assume a standard ~3.5m lane width
            if (!has_left)  left_bound  = right_bound + 3.5f;
            if (!has_right) right_bound = left_bound  - 3.5f;

            // Step 5: Sanity check lane width
            float lane_width = left_bound - right_bound;
            if (lane_width < 2.2f || lane_width > 5.5f) { publish_offset(false); return; }

            // Step 6: Center = midpoint
            float raw_center = (left_bound + right_bound) / 2.0f;
            raw_center = std::clamp(raw_center, -1.5f, 1.5f);
            filtered_offset_ = (alpha_ * raw_center) + ((1.0f - alpha_) * filtered_offset_);

            publish_offset(true);

        } catch (const std::exception& e) { 
            RCLCPP_ERROR(this->get_logger(), "Lane Centering Error: %s", e.what()); 
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SemanticLaneCentering>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}