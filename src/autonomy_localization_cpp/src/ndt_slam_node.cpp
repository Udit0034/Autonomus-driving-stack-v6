#include <memory>
#include <cmath>
#include <vector>
#include <string>
#include <filesystem>
#include <Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl/common/transforms.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h> 

#include <cmath>
#include <cstdlib>
#include <iostream>

// 🎯 FIX FOR -11 SEGFAULTS
typedef Eigen::Matrix<float, 4, 4, Eigen::DontAlign> Matrix4f_Unaligned;

// 🎯 PCL COMPATIBILITY HACK
template <typename PointSource, typename PointTarget>
class SafeNdt : public pcl::NormalDistributionsTransform<PointSource, PointTarget> {
public:
    SafeNdt() = default;

    void setMinPointPerVoxelSafe(int min_points) {
        this->target_cells_.setMinPointPerVoxel(min_points);
    }

    bool hasValidTargetCells() {
        return !this->target_cells_.getLeaves().empty();
    }
};

class NdtSlamNode : public rclcpp::Node {
public:
    NdtSlamNode() : Node("ndt_slam_node"), 
                    ekf_initialized_(false), 
                    map_initialized_(false),
                    slam_pose_(Matrix4f_Unaligned::Identity()),
                    slam_pose_initialized_(false),
                    ekf_trust_distance_(0.0f) {
        
        // Mute PCL terminal spam
        pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

        ekf_guess_ = Matrix4f_Unaligned::Identity();
        last_keyframe_pose_ = Matrix4f_Unaligned::Identity();
        
        // Initialize lever arm transform
        lidar_to_base_ = Eigen::Matrix4f::Identity();
        lidar_to_base_(2, 3) = -2.10f; // LiDAR is 2.1m above base_link, offset points to base frame
        
        target_map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());

        // Tuned Standard SLAM Params
        this->declare_parameter("voxel_size", 0.3);
        this->declare_parameter("ndt_transformation_epsilon", 0.001);
        this->declare_parameter("ndt_step_size", 0.05);
        this->declare_parameter("ndt_resolution", 1.5);
        this->declare_parameter("ndt_max_iterations", 60);
        this->declare_parameter("keyframe_distance", 0.5);

        // Map Persistence & Mode Params
        this->declare_parameter("map_file", "");           
        this->declare_parameter("mapping_mode", true);     
        this->declare_parameter("map_extend_radius", 5.0); 

        // 🎯 ANTI-SIGFPE Parameter Fallbacks 
        // 🚨 CRITICAL FIX: Set NDT parameters BEFORE loading the map to prevent segfaults
        double v_size = this->get_parameter("voxel_size").as_double();
        if (v_size <= 0.01) v_size = 0.3;
        voxel_filter_.setLeafSize(v_size, v_size, v_size);

        double epsilon = this->get_parameter("ndt_transformation_epsilon").as_double();
        if (epsilon <= 0.0) epsilon = 0.001;
        ndt_.setTransformationEpsilon(epsilon);

        double step = this->get_parameter("ndt_step_size").as_double();
        if (step <= 0.0) step = 0.05;
        ndt_.setStepSize(step);

        double res = this->get_parameter("ndt_resolution").as_double();
        if (res <= 0.0) res = 1.5;
        ndt_.setResolution(res);
        
        ndt_.setMaximumIterations(this->get_parameter("ndt_max_iterations").as_int());
        ndt_.setMinPointPerVoxelSafe(6);

        // ✅ MAP LOADING MOVED HERE: Now NDT has valid settings when building the KD-Tree
        map_file_ = this->get_parameter("map_file").as_string();
        mapping_mode_ = this->get_parameter("mapping_mode").as_bool();

        // Load existing map if available
        if (!map_file_.empty() && std::filesystem::exists(map_file_)) {
            pcl::io::loadPCDFile(map_file_, *target_map_cloud_);
            if (!target_map_cloud_->empty()) {
                if (setTargetSafe(target_map_cloud_)) {
                    map_initialized_ = true;
                    RCLCPP_INFO(this->get_logger(), "📂 Loaded existing map: %zu points from %s", 
                                target_map_cloud_->size(), map_file_.c_str());
                }
            }
        }

        // Subscribers & Publishers
        bev_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/dashboard/pred/bev", 10, std::bind(&NdtSlamNode::bevCallback, this, std::placeholders::_1));

        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/carla/hero/lidar_top/point_cloud", rclcpp::SensorDataQoS(), 
            std::bind(&NdtSlamNode::lidarCallback, this, std::placeholders::_1));

        ekf_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ekf/odometry", 10, std::bind(&NdtSlamNode::ekfCallback, this, std::placeholders::_1));

        ndt_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/slam/odometry", 10);
    }

    ~NdtSlamNode() {
        if (mapping_mode_ && !map_file_.empty() && target_map_cloud_ && !target_map_cloud_->empty()) {
            RCLCPP_INFO(this->get_logger(), "💾 Saving map: %zu points to %s", 
                        target_map_cloud_->size(), map_file_.c_str());
            pcl::io::savePCDFileBinary(map_file_, *target_map_cloud_);
        }
    }

private:
    bool setTargetSafe(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
        ndt_.setInputTarget(cloud);
        if (!ndt_.hasValidTargetCells()) {
            RCLCPP_ERROR(this->get_logger(), "NDT target has ZERO valid voxels from %zu points — kdtree_ will be null!", cloud->size());
            return false;
        }
        RCLCPP_DEBUG(this->get_logger(), "NDT target OK from %zu points", cloud->size());
        return true;
    }

    void bevCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        try {
            if (msg->encoding == "mono8" || msg->encoding == "8UC1") {
                latest_bev_ = cv_bridge::toCvCopy(msg, "mono8")->image.clone();
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "BEV cv_bridge exception: %s", e.what());
        }
    }

    void ekfCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        Eigen::Quaternionf q(
            msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y, msg->pose.pose.orientation.z
        );
        q.normalize();
        
        ekf_guess_.block<3,3>(0,0) = q.toRotationMatrix();
        ekf_guess_(0,3) = msg->pose.pose.position.x;
        ekf_guess_(1,3) = msg->pose.pose.position.y;
        ekf_guess_(2,3) = msg->pose.pose.position.z;
        ekf_guess_.row(3) << 0.0f, 0.0f, 0.0f, 1.0f; 
        
        ekf_initialized_ = true;
    }

    template<typename T>
    bool isMatrixValid(const T& mat, std::string& reason) {
        if (!mat.allFinite()) {
            reason = "Contains NaN or Infinite values";
            return false;
        }
        return true;
    }

    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (!ekf_initialized_) {
            RCLCPP_INFO_ONCE(this->get_logger(), "Waiting for EKF Global Anchor... (Muting further messages)");
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *raw_cloud);
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr base_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*raw_cloud, *base_cloud, lidar_to_base_);

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        
        size_t non_finite_count = 0;
        size_t close_proximity_count = 0;

        for (auto pt : base_cloud->points) {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
                non_finite_count++;
                continue;
            }
            
            double sq_dist = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            
            if (sq_dist <= 1.0) {
                close_proximity_count++;
                continue;
            }
            
            if (sq_dist > 2500.0) {
                continue;
            }
            
            if (pt.z < -0.3f) {
                continue; 
            }
            
            bool is_dynamic = false;
            if (!latest_bev_.empty() && latest_bev_.rows == 800 && latest_bev_.cols == 500) {
                int r = static_cast<int>((50.0f - pt.x) * 10.0f);
                int c = static_cast<int>((25.0f - pt.y) * 10.0f);

                if (r >= 0 && r < 800 && c >= 0 && c < 500) {
                    uint8_t sem_class = latest_bev_.at<uint8_t>(r, c);
                    if ((sem_class >= 12 && sem_class <= 19) || sem_class == 21) {
                        is_dynamic = true;
                    }
                }
            }

            if (!is_dynamic) {
                filtered_cloud->points.push_back(pt);
            }   
        }

        if (non_finite_count > 0 || close_proximity_count > 0) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "⚠️ [Data Filter] Discarded Points: %zu Non-Finite (NaN/Inf), %zu Ego-Proximity (<1m)", 
                non_finite_count, close_proximity_count);
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
        voxel_filter_.setInputCloud(filtered_cloud);
        voxel_filter_.filter(*downsampled);

        if (downsampled->empty()) {
            return;
        }

        Matrix4f_Unaligned current_guess;
        if (slam_pose_initialized_) {
            // 1. Calculate how much the EKF says we moved since the last LiDAR frame
            Matrix4f_Unaligned ekf_delta = last_ekf_pose_.inverse() * ekf_guess_;
            
            // 2. Apply that relative movement to our last known good SLAM pose
            current_guess = slam_pose_ * ekf_delta;

            // ✅ BLEND LOGIC MOVED HERE: Safely out of the way of the very first frame
            float ekf_slam_dist = (ekf_guess_.block<3,1>(0,3) - slam_pose_.block<3,1>(0,3)).norm();
            if (ekf_slam_dist < 3.0f) {  // only blend if EKF is within 3m of SLAM
                float alpha = 0.15f;     // 15% EKF, 85% SLAM continuity
                current_guess.block<3,1>(0,3) = 
                    (1.0f - alpha) * slam_pose_.block<3,1>(0,3) + 
                    alpha * ekf_guess_.block<3,1>(0,3);
            }
        } else {
            current_guess = ekf_guess_;  // first frame only: use EKF global anchor
        }
            
        std::string matrix_error_reason;
        if (!isMatrixValid(current_guess, matrix_error_reason)) {
            RCLCPP_ERROR(this->get_logger(), "❌ [Pose Guard] Discarded Frame: Initial transformation guess is invalid! Reason: %s", 
                        matrix_error_reason.c_str());
            return;
        }

        if (!map_initialized_) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::transformPointCloud(*downsampled, *transformed_cloud, current_guess);
            
            for (const auto& pt : transformed_cloud->points) {
                if (pt.z > -0.5f && pt.z < 3.5f) {
                    target_map_cloud_->points.push_back(pt);
                }
            }
            target_map_cloud_->width = target_map_cloud_->points.size();
            target_map_cloud_->height = 1;
            
            if (target_map_cloud_->size() < 50) {
                RCLCPP_WARN(this->get_logger(), "❌ [Map Init] First scan has insufficient features (%zu pts). Waiting for denser data.", target_map_cloud_->size());
                target_map_cloud_->clear();
                return;
            }
            
            if (!setTargetSafe(target_map_cloud_)) {
                RCLCPP_ERROR(this->get_logger(), "❌ [NDT Guard] Discarded Map: Target Cloud failed voxelization.");
                target_map_cloud_->clear();
                return;
            }
            
            last_keyframe_pose_ = current_guess;
            map_initialized_ = true;
            
            slam_pose_ = current_guess;
            last_ekf_pose_ = ekf_guess_;
            slam_pose_initialized_ = true;

            RCLCPP_INFO(this->get_logger(), "🎯 NDT Map Anchored GLOBALLY at X:%.1f Y:%.1f", current_guess(0,3), current_guess(1,3));
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        ndt_.setInputSource(downsampled);
        
        Eigen::Matrix4f align_guess = current_guess;
        align_guess(0,3) += 0.0001f; 

        ndt_.align(*aligned_cloud, align_guess); 

        if (!ndt_.hasConverged()) {
            RCLCPP_WARN(this->get_logger(), "❌ [Registration Fault] NDT optimization failed to converge. Frame dropped.");
            return;
        }

        Eigen::Matrix4f raw_result = ndt_.getFinalTransformation();
        
        double correction_dist = (raw_result.block<3,1>(0,3) - current_guess.block<3,1>(0,3)).norm();
        if (correction_dist > 3.0) { 
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "⚠️ [NDT] Optimization diverged (Jumped %.2fm). Frame dropped.", correction_dist);
            return;
        }
        
        Eigen::Matrix3f R_delta = current_guess.block<3,3>(0,0).transpose() * raw_result.block<3,3>(0,0);
        float yaw_correction = std::atan2(R_delta(1,0), R_delta(0,0));
        if (std::abs(yaw_correction) > 0.5f) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                "⚠️ [NDT] Yaw jump of %.2f rad rejected", yaw_correction);
            return;
        }

        if (!isMatrixValid(raw_result, matrix_error_reason)) {
            RCLCPP_ERROR(this->get_logger(), "❌ [Numerical Guard] Result is unusable! Reason: %s", matrix_error_reason.c_str());
            return;
        }

        Matrix4f_Unaligned final_transform = raw_result;

        slam_pose_ = final_transform;
        last_ekf_pose_ = ekf_guess_; 
        slam_pose_initialized_ = true;

        double delta = (final_transform.block<3,1>(0,3) - last_keyframe_pose_.block<3,1>(0,3)).norm();
        if (delta > this->get_parameter("keyframe_distance").as_double()) {
            
            bool should_extend = mapping_mode_;
            if (!mapping_mode_) {
                should_extend = (delta > this->get_parameter("map_extend_radius").as_double());
            }

            if (should_extend && target_map_cloud_->size() < 500000) {
                for (const auto& pt : aligned_cloud->points) {
                    if (pt.z > -0.5f && pt.z < 3.5f) {
                        target_map_cloud_->points.push_back(pt);
                    }
                }
                
                voxel_filter_.setInputCloud(target_map_cloud_);
                pcl::PointCloud<pcl::PointXYZ>::Ptr new_map(new pcl::PointCloud<pcl::PointXYZ>());
                voxel_filter_.filter(*new_map);

                if (!setTargetSafe(new_map)) {
                    RCLCPP_WARN(this->get_logger(), "⚠️ [Map Append] Map update produced singular matrices. Reverting expansion.");
                    return; 
                }
                target_map_cloud_ = new_map;
            }
            last_keyframe_pose_ = final_transform;
        }

        Eigen::Matrix3f rot_mat = final_transform.block<3, 3>(0, 0);
        Eigen::Quaternionf q(rot_mat);
        q.normalize(); 

        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header.stamp = msg->header.stamp;
        odom_msg.header.frame_id = "map";
        odom_msg.child_frame_id = "base_link";
        odom_msg.pose.pose.position.x = final_transform(0, 3);
        odom_msg.pose.pose.position.y = final_transform(1, 3);
        odom_msg.pose.pose.position.z = final_transform(2, 3);
        odom_msg.pose.pose.orientation.w = q.w();
        odom_msg.pose.pose.orientation.x = q.x();
        odom_msg.pose.pose.orientation.y = q.y();
        odom_msg.pose.pose.orientation.z = q.z();

        ndt_odom_pub_->publish(odom_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr       bev_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       ekf_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr          ndt_odom_pub_;

    cv::Mat latest_bev_;

    pcl::VoxelGrid<pcl::PointXYZ>                                  voxel_filter_;
    SafeNdt<pcl::PointXYZ, pcl::PointXYZ>                          ndt_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr                            target_map_cloud_;

    std::string map_file_;
    bool        mapping_mode_;

    Eigen::Matrix4f lidar_to_base_;

    Matrix4f_Unaligned ekf_guess_;
    bool               ekf_initialized_;
    bool               map_initialized_;
    Matrix4f_Unaligned last_keyframe_pose_;

    Matrix4f_Unaligned slam_pose_;          
    bool               slam_pose_initialized_;            
    float              ekf_trust_distance_;        
    Matrix4f_Unaligned last_ekf_pose_;      
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NdtSlamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}