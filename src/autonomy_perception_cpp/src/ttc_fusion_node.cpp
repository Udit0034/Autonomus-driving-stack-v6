#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <vector>

using json = nlohmann::json;

class TTCFusionNode : public rclcpp::Node {
public:
    TTCFusionNode() : Node("ttc_fusion_node"), v_ego_speed_(0.0) {
        
        // Extrinsics for Front Left Camera
        double r = 0.0, p = -12.0 * M_PI / 180.0, y = 0.0;
        double cy = std::cos(y), sy = std::sin(y);
        double cp = std::cos(p), sp = std::sin(p);
        double cr = std::cos(r), sr = std::sin(r);

        cam_R_ << cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr,
                  sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr,
                  -sp,   cp*sr,            cp*cr;
        cam_T_ << 1.4, -0.25, 1.5;

        // Semantic harmonization mapping
        seg_to_general_ = {
            {12, "pedestrian"}, {13, "pedestrian"}, {14, "vehicle"}, 
            {15, "vehicle"}, {16, "vehicle"}, {17, "vehicle"}, 
            {18, "vehicle"}, {19, "vehicle"}
        };

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ekf/odometry", 10, std::bind(&TTCFusionNode::odom_callback, this, std::placeholders::_1));
        
        radar_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/perception/radar/dynamic_targets", 10, std::bind(&TTCFusionNode::radar_callback, this, std::placeholders::_1));
        
        vision_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/inference/front_left/objects", 10, std::bind(&TTCFusionNode::vision_callback, this, std::placeholders::_1));

        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/inference/front_left/depth", 10, std::bind(&TTCFusionNode::depth_callback, this, std::placeholders::_1));
            
        seg_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/inference/front_left/seg", 10, std::bind(&TTCFusionNode::seg_callback, this, std::placeholders::_1));

        fused_pub_ = this->create_publisher<std_msgs::msg::String>("/planning/fused_tracked_objects", 10);

        // Timer for steady state publishing at 20Hz (50ms)
        publish_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&TTCFusionNode::publish_last_fused, this)
        );

        RCLCPP_INFO(this->get_logger(), "🧠 Full-Fidelity C++ TTC Semantic Fusion Node (Seg-Primary) Online.");
    }

private:
    double v_ego_speed_;
    std::vector<std::array<double, 3>> radar_targets_;
    Eigen::Matrix3d cam_R_;
    Eigen::Vector3d cam_T_;
    std::unordered_map<int, std::string> seg_to_general_;
    
    // State persistence for the 20Hz publisher
    json last_fused_output_ = json::array();

    // Image buffers and mutex
    cv::Mat latest_depth_;   // 32FC1 float depth map
    cv::Mat latest_seg_;     // 8UC1 segmentation label map
    std::mutex img_mutex_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr radar_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr vision_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr seg_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr fused_pub_;
    rclcpp::TimerBase::SharedPtr publish_timer_;

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        v_ego_speed_ = std::sqrt(std::pow(msg->twist.twist.linear.x, 2) + std::pow(msg->twist.twist.linear.y, 2));
    }

    void radar_callback(const std_msgs::msg::String::SharedPtr msg) {
        radar_targets_.clear();
        try {
            auto j = json::parse(msg->data);
            for (const auto& item : j) {
                radar_targets_.push_back({item[0].get<double>(), item[1].get<double>(), item[2].get<double>()});
            }
        } catch (...) { }
    }

    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(img_mutex_);
        try {
            latest_depth_ = cv_bridge::toCvCopy(msg, "32FC1")->image;
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void seg_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(img_mutex_);
        try {
            latest_seg_ = cv_bridge::toCvCopy(msg, "mono8")->image;
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void vision_callback(const std_msgs::msg::String::SharedPtr msg) {
        // Capture latest images safely
        cv::Mat seg, depth;
        {
            std::lock_guard<std::mutex> lock(img_mutex_);
            if (latest_seg_.empty() || latest_depth_.empty()) return;
            seg = latest_seg_.clone();
            depth = latest_depth_.clone();
        }

        try {
            auto payload = json::parse(msg->data);
            if (!payload.contains("camera_meta")) return;

            json fused_output = json::array();

            double fx = payload["camera_meta"]["intrinsics"]["fx"].get<double>();
            double cx = payload["camera_meta"]["intrinsics"]["cx"].get<double>();
            double fy = payload["camera_meta"]["intrinsics"]["fy"].get<double>();
            double cy = payload["camera_meta"]["intrinsics"]["cy"].get<double>();

            // ==============================================================================
            // Step A: Seg-based blob detection (Replaces YOLO as primary detector)
            // ==============================================================================
            cv::Mat mask = cv::Mat::zeros(seg.size(), CV_8UC1);
            
            for (int r = 0; r < seg.rows; ++r) {
                for (int c = 0; c < seg.cols; ++c) {
                    uint8_t val = seg.at<uint8_t>(r, c);
                    if (val >= 12 && val <= 19) {
                        mask.at<uint8_t>(r, c) = 255;
                    }
                }
            }

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (const auto& contour : contours) {
                if (cv::contourArea(contour) < 200.0) continue;

                cv::Rect bbox = cv::boundingRect(contour);
                
                int u_center = bbox.x + bbox.width / 2;
                int v_center = bbox.y + bbox.height / 2;
                
                u_center = std::clamp(u_center, 0, depth.cols - 1);
                v_center = std::clamp(v_center, 0, depth.rows - 1);

                float z_depth = depth.at<float>(v_center, u_center);
                
                // 🎯 PROBLEM 2 FIX: Add depth sanity check before projecting
                if (std::isnan(z_depth) || z_depth <= 0.5f || z_depth > 80.0f) continue;

                std::unordered_map<uint8_t, int> class_counts;
                for (int r = bbox.y; r < bbox.y + bbox.height; ++r) {
                    for (int c = bbox.x; c < bbox.x + bbox.width; ++c) {
                        if (r >= 0 && r < seg.rows && c >= 0 && c < seg.cols) {
                            uint8_t val = seg.at<uint8_t>(r, c);
                            if (val >= 12 && val <= 19) class_counts[val]++;
                        }
                    }
                }

                uint8_t majority_class = 0;
                int max_count = 0;
                for (const auto& [cls, count] : class_counts) {
                    if (count > max_count) {
                        max_count = count;
                        majority_class = cls;
                    }
                }

                std::string cls_name = seg_to_general_.count(majority_class) ? seg_to_general_[majority_class] : "unknown";
                int track_id = -1;

                // ==============================================================================
                // Step B: YOLO as class confirmation and tracker ID assignment
                // ==============================================================================
                if (payload.contains("detections")) {
                    double best_iou = 0.0;
                    for (const auto& yolo_obj : payload["detections"]) {
                        auto yolo_bbox = yolo_obj["bbox"];
                        cv::Rect yolo_rect(
                            yolo_bbox[0].get<int>(),
                            yolo_bbox[1].get<int>(),
                            yolo_bbox[2].get<int>() - yolo_bbox[0].get<int>(),
                            yolo_bbox[3].get<int>() - yolo_bbox[1].get<int>()
                        );

                        cv::Rect intersection = bbox & yolo_rect;
                        double inter_area = intersection.area();
                        double union_area = bbox.area() + yolo_rect.area() - inter_area;
                        double iou = (union_area > 0) ? (inter_area / union_area) : 0.0;

                        if (iou > 0.3 && iou > best_iou) {
                            best_iou = iou;
                            cls_name = yolo_obj.value("class_name", cls_name);
                            track_id = yolo_obj.value("track_id", -1);
                        }
                    }
                }

                // ==============================================================================
                // Step C: Back-projection and Radar Association
                // ==============================================================================
                double x_cam = (u_center - cx) * (z_depth / fx);
                double y_cam = (v_center - cy) * (z_depth / fy);
                Eigen::Vector3d p_cam(x_cam, y_cam, z_depth);

                Eigen::Vector3d p_ego = (cam_R_ * p_cam) + cam_T_;
                double ego_x = p_ego.x(), ego_y = p_ego.y();

                // 🎯 PROBLEM 2 FIX: Add ego-frame coordinates validation
                if (ego_x < -2.0 || ego_x > 60.0) continue;
                if (std::abs(ego_y) > 8.0) continue; 

                double assigned_speed = 0.0;
                double min_dist = 6.0; 
                bool matched_radar = false;

                for (const auto& r_target : radar_targets_) {
                    double dist = std::sqrt(std::pow(ego_x - r_target[0], 2) + std::pow(ego_y - r_target[1], 2));
                    if (dist < min_dist) {
                        min_dist = dist;
                        assigned_speed = r_target[2];
                        matched_radar = true;
                    }
                }

                double distance_to_obj = std::sqrt(ego_x*ego_x + ego_y*ego_y);
                double ttc = 999.0;

                if (matched_radar) {
                    double rel_speed = -assigned_speed;
                    ttc = (rel_speed > 0.1) ? (distance_to_obj / rel_speed) : 999.0;
                } else {
                    // Step D: Fallback TTC when no radar associates
                    if (min_dist >= 6.0 && distance_to_obj < 30.0 && ego_x > 1.0) {
                        assigned_speed = -v_ego_speed_ * 0.8; 
                        if (v_ego_speed_ > 0.1) {
                            ttc = distance_to_obj / v_ego_speed_;
                        }
                    }
                }

                fused_output.push_back({
                    std::round(ego_x * 100.0) / 100.0,
                    std::round(ego_y * 100.0) / 100.0,
                    std::round(assigned_speed * 100.0) / 100.0,
                    std::round(ttc * 100.0) / 100.0,
                    track_id,
                    cls_name
                });
            } // End of camera-based detection loop

            // ==============================================================================
            // 🎯 PROBLEM 1 FIX: Recover Radar-Only Targets (Unmatched Side/Rear Clusters)
            // ==============================================================================
            std::vector<bool> radar_matched(radar_targets_.size(), false);
            
            for (const auto& fused_obj : fused_output) {
                double ox = fused_obj[0].get<double>();
                double oy = fused_obj[1].get<double>();
                for (size_t ri = 0; ri < radar_targets_.size(); ++ri) {
                    double dist = std::hypot(ox - radar_targets_[ri][0],
                                             oy - radar_targets_[ri][1]);
                    if (dist < 6.0) {
                        radar_matched[ri] = true;
                    }
                }
            }
            
            for (size_t ri = 0; ri < radar_targets_.size(); ++ri) {
                if (radar_matched[ri]) continue;
                double rx = radar_targets_[ri][0];
                double ry = radar_targets_[ri][1];
                double rv = radar_targets_[ri][2];
                
                double rdist = std::hypot(rx, ry);
                if (rdist > 50.0 || rdist < 1.0) continue;
                
                double rel_speed = -rv;  
                double ttc = (rel_speed > 0.1) ? (rdist / rel_speed) : 999.0;
                
                fused_output.push_back({
                    std::round(rx * 100.0) / 100.0,
                    std::round(ry * 100.0) / 100.0,
                    std::round(rv * 100.0) / 100.0,
                    std::round(ttc * 100.0) / 100.0,
                    -1,          // track_id = -1 signifies radar-only targets
                    "vehicle"    
                });
            }

            // Update local state for the 20Hz steady-state output timer
            last_fused_output_ = fused_output;

        } catch (...) { }
    }

    void publish_last_fused() {
        std_msgs::msg::String output_msg;
        output_msg.data = last_fused_output_.dump();
        fused_pub_->publish(output_msg);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TTCFusionNode>());
    rclcpp::shutdown();
    return 0;
}