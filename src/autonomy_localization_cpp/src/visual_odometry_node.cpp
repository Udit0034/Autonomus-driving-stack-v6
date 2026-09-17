#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

// NEW PCL INCLUDES FOR LIDAR DEPTH
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <mutex>
#include <map>
#include <vector>
#include <cmath>
#include <memory>
#include <limits>
#include <algorithm>
#include <iostream> 

// =============================================================================
// EGO-COMPENSATED RGB-D VISUAL ODOMETRY (LIDAR FUSION UPGRADED)
// =============================================================================
class VisualOdometry {
private:
    int     width_, height_;
    double fx_, fy_, cx_, cy_;
    cv::Mat K_, dist_coeffs_;
    Eigen::Vector3d lever_arm_;
    Eigen::Matrix3d R_cam_to_body_;
    
    // LiDAR transform state
    Eigen::Matrix4d T_lidar_to_cam_;

    std::vector<cv::Mat>     prev_pyr_;
    std::vector<float>         prev_kps_depth_;
    std::vector<cv::Point2f> prev_kps_;

    // Thread-safe state variables
    Eigen::Matrix3d R_ego_world_;
    Eigen::Vector3d t_ego_world_;
    std::mutex      state_mutex_;

    const int    max_corners_   = 300;
    const double quality_level_ = 0.01;
    const double min_distance_  = 15.0;
    const int    block_size_    = 3;
    const cv::Size win_size_    = cv::Size(15, 15); 
    const int max_level_        = 2;                    
    const cv::TermCriteria lk_criteria_ = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.01);
    const cv::TermCriteria subpix_criteria_ = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.01);

public:
    bool initialized = false;

    VisualOdometry(double fov = 90.0, int width = 1280, int height = 720)
        : width_(width), height_(height)
    {
        fx_ = (width_  / 2.0) / std::tan((fov * M_PI / 180.0) / 2.0);
        fy_ = fx_;
        cx_ = width_  / 2.0;
        cy_ = height_ / 2.0;

        K_ = (cv::Mat_<float>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
        dist_coeffs_ = cv::Mat::zeros(4, 1, CV_32F);

        lever_arm_ = Eigen::Vector3d(1.4, -0.25, 1.5); 

        // Fix 1: Pitch rotation (Updated to -11 deg for precise cam to veh match)
        double pitch_rad = -11.0 * M_PI / 180.0;
        double cp = std::cos(pitch_rad), sp = std::sin(pitch_rad);
        Eigen::Matrix3d R_pitch;
        R_pitch << 1,  0,   0,
                   0, cp, -sp,
                   0, sp,  cp;

        Eigen::Matrix3d R_axes;
        R_axes <<  0,  0,  1,
                  -1,  0,  0,
                   0, -1,  0;

        R_cam_to_body_ = R_axes * R_pitch;

        R_ego_world_ = Eigen::Matrix3d::Identity();
        t_ego_world_ = Eigen::Vector3d::Zero();

        // 🚀 NEW: Build T_cam_to_veh (4x4)
        Eigen::Matrix4d T_cam_to_veh = Eigen::Matrix4d::Identity();
        T_cam_to_veh.block<3,3>(0,0) = R_cam_to_body_.transpose(); // R_body_to_cam = R_cam_to_body^T
        T_cam_to_veh(0,3) = 1.4;
        T_cam_to_veh(1,3) = -0.25;
        T_cam_to_veh(2,3) = 1.5;

        // 🚀 NEW: Build T_lidar_to_vehicle (4x4) — pure translation, no rotation
        Eigen::Matrix4d T_lidar_to_veh = Eigen::Matrix4d::Identity();
        T_lidar_to_veh(2,3) = 2.10;

        // 🚀 NEW: T_lidar_to_cam = T_cam_to_veh^-1 * T_lidar_to_veh
        T_lidar_to_cam_ = T_cam_to_veh.inverse() * T_lidar_to_veh;
    }

    int get_width() const { return width_; }
    int get_height() const { return height_; }

    void set_gnss_anchor(const Eigen::Vector3d& gnss_t, double gnss_yaw) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        
        t_ego_world_ = gnss_t;

        double vo_pitch = -std::asin(-R_ego_world_(2,0));
        double vo_roll  =  std::atan2(R_ego_world_(2,1), R_ego_world_(2,2));

        Eigen::Matrix3d R_z = Eigen::AngleAxisd(gnss_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        Eigen::Matrix3d R_y = Eigen::AngleAxisd(vo_pitch, Eigen::Vector3d::UnitY()).toRotationMatrix();
        Eigen::Matrix3d R_x = Eigen::AngleAxisd(vo_roll,  Eigen::Vector3d::UnitX()).toRotationMatrix();
        
        R_ego_world_ = R_z * R_y * R_x;
        initialized = true;
    }

    // 🚀 NEW: Project LiDAR points to synthetic Depth Map
    void project_lidar_to_depth(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, cv::Mat& out_depth) {
        for (const auto& pt : cloud->points) {
            // Skip invalid/nan points
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;

            // Transform point from LiDAR frame to camera frame
            Eigen::Vector4d pt_lidar(pt.x, pt.y, pt.z, 1.0);
            Eigen::Vector4d pt_cam = T_lidar_to_cam_ * pt_lidar;

            // CARLA LiDAR is right-handed, camera is OpenCV convention:
            double cam_x =  pt_cam(1);   // after transform: ego-Y maps to cam-X
            double cam_y = -pt_cam(2);   // ego-Z maps to -cam-Y  
            double cam_z =  pt_cam(0);   // ego-X (forward) maps to depth

            // Only points in front of camera
            if (cam_z < 0.5 || cam_z > 80.0) continue;

            // Project to pixel
            double u = fx_ * cam_x / cam_z + cx_;
            double v = fy_ * cam_y / cam_z + cy_;

            int iu = static_cast<int>(std::round(u));
            int iv = static_cast<int>(std::round(v));

            if (iu < 0 || iu >= width_ || iv < 0 || iv >= height_) continue;

            // Keep the CLOSER depth if two points project to the same pixel
            float existing = out_depth.at<float>(iv, iu);
            if (existing == 0.0f || static_cast<float>(cam_z) < existing) {
                out_depth.at<float>(iv, iu) = static_cast<float>(cam_z);
            }
        }
    }

    // 🚀 NEW: Small window search prioritizing LiDAR depth over Stereonet
    float get_best_depth(const cv::Mat& fused_depth, const cv::Mat& lidar_depth, int px, int py, int search_radius = 3) {
        // First pass: look for a LiDAR hit within search_radius
        float best_lidar = 0.0f;
        float best_lidar_dist = float(search_radius * search_radius + 1);

        int h = lidar_depth.rows, w = lidar_depth.cols;
        for (int dy = -search_radius; dy <= search_radius; ++dy) {
            for (int dx = -search_radius; dx <= search_radius; ++dx) {
                int nx = px + dx, ny = py + dy;
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                float d = lidar_depth.at<float>(ny, nx);
                if (d > 0.5f) {
                    float dist2 = float(dx*dx + dy*dy);
                    if (dist2 < best_lidar_dist) {
                        best_lidar_dist = dist2;
                        best_lidar = d;
                    }
                }
            }
        }

        if (best_lidar > 0.5f) return best_lidar;  // LiDAR wins

        // Fallback: use fused depth (stereonet) at this exact pixel
        float sd = fused_depth.at<float>(py, px);
        return sd;
    }

    // UPDATED SIGNATURE: Takes raw LiDAR depth specifically for localized window search
    std::vector<double> process_frame(const cv::Mat& rgb, const cv::Mat& fused_depth, const cv::Mat& lidar_depth, double pred_dist) {
        cv::Mat curr_gray;
        cv::cvtColor(rgb, curr_gray, cv::COLOR_BGR2GRAY);

        if (!initialized) {
            return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; 
        }

        float scale_x = (float)fused_depth.cols / width_;
        float scale_y = (float)fused_depth.rows / height_;

        if ((int)prev_kps_.size() < 30) {
            cv::goodFeaturesToTrack(curr_gray, prev_kps_, max_corners_, quality_level_, min_distance_, cv::Mat(), block_size_);
            if (!prev_kps_.empty()) {
                cv::cornerSubPix(curr_gray, prev_kps_, cv::Size(5, 5), cv::Size(-1, -1), subpix_criteria_);
            }

            cv::buildOpticalFlowPyramid(curr_gray, prev_pyr_, win_size_, max_level_, true);

            prev_kps_depth_.clear();
            for (const auto& pt : prev_kps_) {
                int dx = std::clamp(static_cast<int>(pt.x * scale_x), 0, fused_depth.cols - 1);
                int dy = std::clamp(static_cast<int>(pt.y * scale_y), 0, fused_depth.rows - 1);
                // 🚀 NEW: Lookup using localized prioritized window search
                prev_kps_depth_.push_back(get_best_depth(fused_depth, lidar_depth, dx, dy, 3));
            }

            return get_pose_tuple();
        }

        std::vector<cv::Mat> curr_pyr;
        cv::buildOpticalFlowPyramid(curr_gray, curr_pyr, win_size_, max_level_, true);

        std::vector<cv::Point2f> curr_kps;
        std::vector<uchar>       status_fwd;
        std::vector<float>       err;
        cv::calcOpticalFlowPyrLK(prev_pyr_, curr_pyr, prev_kps_, curr_kps, status_fwd, err, win_size_, max_level_, lk_criteria_);

        std::vector<cv::Point2f> kps_back;
        std::vector<uchar>       status_bwd;
        cv::calcOpticalFlowPyrLK(curr_pyr, prev_pyr_, curr_kps, kps_back, status_bwd, err, win_size_, max_level_, lk_criteria_);

        std::vector<cv::Point3f> obj_pts;
        std::vector<cv::Point2f> img_pts;
        std::vector<cv::Point2f> good_new;
        std::vector<float>       good_new_depth;

        for (size_t i = 0; i < status_fwd.size(); ++i) {
            if (status_fwd[i] && status_bwd[i]) {
                double diff = cv::norm(prev_kps_[i] - kps_back[i]);
                if (diff < 1.0) {
                    const cv::Point2f& old_pt = prev_kps_[i];
                    const cv::Point2f& new_pt = curr_kps[i];

                    int ix = static_cast<int>(old_pt.x);
                    int iy = static_cast<int>(old_pt.y);
                    if (ix < 0 || ix >= width_ || iy < 0 || iy >= height_) continue;

                    float z = prev_kps_depth_[i];
                    if (z < 1.5f || z > 80.0f) continue;

                    float x3d = (old_pt.x - cx_) * z / fx_;
                    float y3d = (old_pt.y - cy_) * z / fy_;

                    obj_pts.emplace_back(x3d, y3d, z);
                    img_pts.push_back(new_pt);
                    good_new.push_back(new_pt);

                    int dx = std::clamp(static_cast<int>(new_pt.x * scale_x), 0, fused_depth.cols - 1);
                    int dy = std::clamp(static_cast<int>(new_pt.y * scale_y), 0, fused_depth.rows - 1);
                    // 🚀 NEW: Ensure next iteration also uses highest quality depth
                    good_new_depth.push_back(get_best_depth(fused_depth, lidar_depth, dx, dy, 3));
                }
            }
        }

        if ((int)obj_pts.size() >= 10) {
            cv::Mat rvec, tvec;
            bool success = cv::solvePnPRansac(
                obj_pts, img_pts, K_, dist_coeffs_,
                rvec, tvec, false, 80, 2.0f, 0.99, cv::noArray(), cv::SOLVEPNP_EPNP);

            if (success) {
                cv::Mat R_mat;
                cv::Rodrigues(rvec, R_mat);

                Eigen::Matrix3d R_eigen;
                Eigen::Vector3d t_eigen;
                for (int i = 0; i < 3; ++i) {
                    t_eigen(i) = tvec.at<double>(i);
                    for (int j = 0; j < 3; ++j) R_eigen(i,j) = R_mat.at<double>(i,j);
                }

                // 🎯 FIXED: Removed .transpose() to invert the mirrored rotation matrix
                Eigen::Vector3d delta_t_cam = -R_eigen * t_eigen;
                Eigen::Matrix3d R_rel_cam   = R_eigen;

                Eigen::Vector3d delta_t_body = R_cam_to_body_ * delta_t_cam;
                Eigen::Matrix3d R_rel_body   = R_cam_to_body_ * R_rel_cam * R_cam_to_body_.transpose();

                double norm_vo = delta_t_body.norm();
                if (norm_vo > 1e-6) {
                    delta_t_body = (delta_t_body / norm_vo) * pred_dist;
                } else {
                    delta_t_body = Eigen::Vector3d(pred_dist, 0.0, 0.0);
                }

                Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
                Eigen::Vector3d delta_t_ego = delta_t_body + (I - R_rel_body) * lever_arm_;

                if (delta_t_ego.norm() > 4.0) {
                    std::cerr << "[VO Filter] Rejected outlier update! Delta norm: " << delta_t_ego.norm() << " meters.\n";
                } else {
                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        t_ego_world_ += R_ego_world_ * delta_t_ego;
                        R_ego_world_  = R_ego_world_ * R_rel_body;
                    }
                }
            }
        }

        if ((int)good_new.size() < 30) {
            prev_kps_.clear(); 
            prev_kps_depth_.clear();
        } else {
            prev_kps_ = good_new;
            prev_kps_depth_ = good_new_depth;
        }

        prev_pyr_ = std::move(curr_pyr);
        return get_pose_tuple();
    }

    std::vector<double> get_pose_tuple() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        double x = t_ego_world_(0);
        double y = t_ego_world_(1);
        double z = t_ego_world_(2);

        double yaw   = std::atan2(R_ego_world_(1,0), R_ego_world_(0,0));
        double pitch = -std::asin(-R_ego_world_(2,0));
        double roll  = std::atan2(R_ego_world_(2,1), R_ego_world_(2,2));

        return {x, y, z, roll, pitch, yaw};
    }
};

// =============================================================================
// ROS2 NODE WRAPPER
// =============================================================================
class VisualOdometryNode : public rclcpp::Node {
private:
    VisualOdometry vo_;
    int  frame_counter_ = 0;
    
    // GNSS & Velocity State
    sensor_msgs::msg::NavSatFix::SharedPtr latest_gnss_front_;
    sensor_msgs::msg::NavSatFix::SharedPtr latest_gnss_rear_;
    bool has_origin_ = false;
    double origin_lat_, origin_lon_, origin_alt_;
    double latest_speed_ = 0.0;
    rclcpp::Time last_frame_time_;

    // 🚀 NEW: Sparse LiDAR depth cache variables
    cv::Mat lidar_depth_map_;
    rclcpp::Time lidar_depth_stamp_;
    std::mutex lidar_depth_mutex_;
    bool lidar_depth_ready_ = false;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr          odom_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr       rgb_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr       depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_; // 🚀 NEW: LiDAR subscription
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       speed_sub_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr   gnss_front_sub_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr   gnss_rear_sub_;

    std::map<double, sensor_msgs::msg::Image::SharedPtr> rgb_cache_;
    std::mutex msg_mutex_;

    const double EARTH_RADIUS = 6378137.0;

public:
    VisualOdometryNode() : Node("visual_odometry_node"), vo_(90.0, 1280, 720) {
        // 🚀 NEW: Added Fusion Parameters
        this->declare_parameter("lidar_depth_max_age_s",  0.15);  
        this->declare_parameter("lidar_search_radius_px", 3);       
        this->declare_parameter("lidar_depth_min_m",      0.5);     
        this->declare_parameter("lidar_depth_max_m",      80.0);    
        this->declare_parameter("depth_fusion_enabled",   true);

        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/vo/odometry", 10);

        auto qos_sensor = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();

        rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/carla/hero/front_left/image", qos_sensor,
            std::bind(&VisualOdometryNode::rgb_callback, this, std::placeholders::_1));

        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/inference/front_left/depth", qos_sensor,
            std::bind(&VisualOdometryNode::depth_callback, this, std::placeholders::_1));

        // 🚀 NEW: Initialize LiDAR point cloud sub using standard sensor QoS
        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/carla/hero/lidar_top/point_cloud", rclcpp::SensorDataQoS(),
            std::bind(&VisualOdometryNode::lidar_callback, this, std::placeholders::_1));

        speed_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/carla/hero/odometry", rclcpp::SystemDefaultsQoS(),
            std::bind(&VisualOdometryNode::speed_callback, this, std::placeholders::_1));

        gnss_front_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
            "/carla/hero/gnss_front", qos_sensor,
            std::bind(&VisualOdometryNode::gnss_front_callback, this, std::placeholders::_1));

        gnss_rear_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
            "/carla/hero/gnss_rear", qos_sensor,
            std::bind(&VisualOdometryNode::gnss_rear_callback, this, std::placeholders::_1));

        last_frame_time_ = this->now();
        RCLCPP_INFO(this->get_logger(), "L4 Visual Odometry Node Active (LiDAR Fusion + Motion Model).");
    }

private:
    void speed_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(msg_mutex_);
        double vx = msg->twist.twist.linear.x;
        double vy = msg->twist.twist.linear.y;
        double vz = msg->twist.twist.linear.z;
        latest_speed_ = std::sqrt(vx*vx + vy*vy + vz*vz);
    }

    void gnss_front_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(msg_mutex_);
        latest_gnss_front_ = msg;
    }

    void gnss_rear_callback(const sensor_msgs::msg::NavSatFix::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(msg_mutex_);
        latest_gnss_rear_ = msg;
    }
    
    // 🚀 NEW: LiDAR Processing Callback
    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        cv::Mat depth_map = cv::Mat::zeros(vo_.get_height(), vo_.get_width(), CV_32F);
        vo_.project_lidar_to_depth(cloud, depth_map);

        std::lock_guard<std::mutex> lock(lidar_depth_mutex_);
        lidar_depth_map_ = depth_map;
        lidar_depth_stamp_ = msg->header.stamp;
        lidar_depth_ready_ = true;
    }

    Eigen::Vector3d geo_to_enu(double lat, double lon, double alt) {
        double x = (lon - origin_lon_) * (M_PI / 180.0) * EARTH_RADIUS * std::cos(origin_lat_ * M_PI / 180.0);
        double y = (lat - origin_lat_) * (M_PI / 180.0) * EARTH_RADIUS;
        double z = alt - origin_alt_;
        return Eigen::Vector3d(x, y, z);
    }

    void process_gnss_snap() {
        std::lock_guard<std::mutex> lock(msg_mutex_);
        if (!latest_gnss_front_ || !latest_gnss_rear_) return;

        if (!has_origin_) {
            origin_lat_ = 0.0;
            origin_lon_ = 0.0;
            origin_alt_ = 0.0;
            has_origin_ = true;
        }

        Eigen::Vector3d f_loc = geo_to_enu(latest_gnss_front_->latitude, latest_gnss_front_->longitude, latest_gnss_front_->altitude);
        Eigen::Vector3d r_loc = geo_to_enu(latest_gnss_rear_->latitude, latest_gnss_rear_->longitude, latest_gnss_rear_->altitude);

        Eigen::Vector3d gnss_t((f_loc.x() + r_loc.x()) / 2.0, 
                               (f_loc.y() + r_loc.y()) / 2.0, 
                               ((f_loc.z() + r_loc.z()) / 2.0) - 1.5);

        double dx = f_loc.x() - r_loc.x();
        double dy = f_loc.y() - r_loc.y();
        double gnss_yaw = std::atan2(dy, dx);

        vo_.set_gnss_anchor(gnss_t, gnss_yaw);
    }

    void rgb_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        double t_key = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        std::lock_guard<std::mutex> lock(msg_mutex_);
        rgb_cache_[t_key] = msg;
        while (rgb_cache_.size() > 5) rgb_cache_.erase(rgb_cache_.begin());
    }

    void depth_callback(const sensor_msgs::msg::Image::SharedPtr depth_msg) {
        sensor_msgs::msg::Image::SharedPtr best_rgb = nullptr;
        
        {
            std::lock_guard<std::mutex> lock(msg_mutex_);
            if (rgb_cache_.empty()) return;

            double depth_time = depth_msg->header.stamp.sec + depth_msg->header.stamp.nanosec * 1e-9;
            double best_key = -1.0, best_dt = std::numeric_limits<double>::max();

            for (const auto& [t_key, rgb_msg] : rgb_cache_) {
                double dt = std::abs(depth_time - t_key);
                if (dt < best_dt) { best_dt = dt; best_key = t_key; best_rgb = rgb_msg; }
            }
            if (best_key < 0 || best_dt > 0.1) return; 
            rgb_cache_.erase(best_key);
        }

        // 🚀 NEW: EKF ONE-SHOT INITIALIZATION (No more periodic snapping)
        if (!vo_.initialized) {
            process_gnss_snap();
            if (!vo_.initialized) return; // Keep returning until successful startup anchor
        }

        try {
            cv_bridge::CvImageConstPtr cv_rgb       = cv_bridge::toCvShare(best_rgb, "bgr8");
            cv_bridge::CvImageConstPtr cv_depth_ptr = cv_bridge::toCvShare(depth_msg, "32FC1");

            cv::Mat working_depth = cv_depth_ptr->image;
            rclcpp::Time current_time = depth_msg->header.stamp;

            // 🚀 NEW: LIDAR-STEREONET DEPTH FUSION
            cv::Mat fused_depth;
            cv::Mat local_lidar_map; // Passed into VO for search prioritizing
            {
                std::lock_guard<std::mutex> lock(lidar_depth_mutex_);
                double lidar_age = (current_time - lidar_depth_stamp_).seconds();
                bool enabled = this->get_parameter("depth_fusion_enabled").as_bool();
                
                if (enabled && lidar_depth_ready_ && lidar_age < this->get_parameter("lidar_depth_max_age_s").as_double()) {
                    // Start with stereonet as base, overlay LiDAR
                    fused_depth = working_depth.clone();   
                    local_lidar_map = lidar_depth_map_.clone();
                    
                    // Overwrite wherever LiDAR has a hit
                    for (int row = 0; row < fused_depth.rows; ++row) {
                        for (int col = 0; col < fused_depth.cols; ++col) {
                            float lid = lidar_depth_map_.at<float>(row, col);
                            if (lid > 0.5f) { 
                                fused_depth.at<float>(row, col) = lid;
                            }
                        }
                    }
                } else {
                    fused_depth = working_depth;
                    local_lidar_map = cv::Mat::zeros(working_depth.size(), CV_32F);
                    if (enabled && lidar_depth_ready_ && lidar_age >= this->get_parameter("lidar_depth_max_age_s").as_double()) {
                        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                            "⚠️ [Depth Fusion] LiDAR depth stale (%.3fs old). Using stereonet only.", lidar_age);
                    }
                }
            }

            double dt = (current_time - last_frame_time_).seconds();
            if (dt <= 0.0 || dt > 0.5) dt = 0.05; 
            last_frame_time_ = current_time;
            
            double pred_dist = latest_speed_ * dt;

            // 🚀 NEW: Passing fused stereonet map + raw LiDAR map to VO node
            std::vector<double> pose = vo_.process_frame(cv_rgb->image, fused_depth, local_lidar_map, pred_dist);

            publish_odometry(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], depth_msg->header.stamp);
            ++frame_counter_;

        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "VO error: %s", e.what());
        }
    }

    void publish_odometry(double x, double y, double z, double roll, double pitch, double yaw, const rclcpp::Time& ts) {
        auto msg = nav_msgs::msg::Odometry();
        msg.header.stamp    = ts;
        msg.header.frame_id = "map";
        msg.child_frame_id  = "base_link";

        msg.pose.pose.position.x = x;
        msg.pose.pose.position.y = y;
        msg.pose.pose.position.z = z;

        Eigen::Quaterniond q =
            Eigen::AngleAxisd(yaw,   Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX());

        msg.pose.pose.orientation.x = q.x();
        msg.pose.pose.orientation.y = q.y();
        msg.pose.pose.orientation.z = q.z();
        msg.pose.pose.orientation.w = q.w();

        odom_pub_->publish(msg);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VisualOdometryNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    try { executor.spin(); } catch (const std::exception& e) {}
    rclcpp::shutdown();
    return 0;
}