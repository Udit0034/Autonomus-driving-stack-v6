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

#include <mutex>
#include <map>
#include <vector>
#include <cmath>
#include <memory>
#include <limits>

// =============================================================================
// EGO-COMPENSATED RGB-D VISUAL ODOMETRY (L4 FUSION UPGRADED)
// =============================================================================
class VisualOdometry {
private:
    int     width_, height_;
    double fx_, fy_, cx_, cy_;
    cv::Mat K_, dist_coeffs_;
    Eigen::Vector3d lever_arm_;
    Eigen::Matrix3d R_cam_to_body_;

    cv::Mat                  prev_gray_;
    cv::Mat                  prev_depth_;
    std::vector<cv::Point2f> prev_kps_;

    // Thread-safe state variables
    Eigen::Matrix3d R_ego_world_;
    Eigen::Vector3d t_ego_world_;
    std::mutex      state_mutex_;

    const int    max_corners_   = 300;
    const double quality_level_ = 0.01;
    const double min_distance_  = 15.0;
    const int    block_size_    = 3;
    const cv::Size win_size_    = cv::Size(21, 21); 
    const int max_level_        = 3;                    
    const cv::TermCriteria lk_criteria_ = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    const cv::TermCriteria subpix_criteria_ = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

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

        lever_arm_ = Eigen::Vector3d(1.4, 0.25, 1.5); // Fixed for standard right-handed

        // Fix 1: Pitch rotation
        double pitch_rad = -12.0 * M_PI / 180.0;
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
    }

    void set_gnss_anchor(const Eigen::Vector3d& gnss_t, double gnss_yaw) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        
        t_ego_world_ = gnss_t;

        // Keep current VO Pitch and Roll, overwrite Yaw
        double vo_pitch = -std::asin(-R_ego_world_(2,0));
        double vo_roll  =  std::atan2(R_ego_world_(2,1), R_ego_world_(2,2));

        Eigen::Matrix3d R_z = Eigen::AngleAxisd(gnss_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        Eigen::Matrix3d R_y = Eigen::AngleAxisd(vo_pitch, Eigen::Vector3d::UnitY()).toRotationMatrix();
        Eigen::Matrix3d R_x = Eigen::AngleAxisd(vo_roll,  Eigen::Vector3d::UnitX()).toRotationMatrix();
        
        R_ego_world_ = R_z * R_y * R_x;
        initialized = true;
    }

    std::vector<double> process_frame(const cv::Mat& rgb, const cv::Mat& depth, double pred_dist) {
        cv::Mat curr_gray;
        cv::cvtColor(rgb, curr_gray, cv::COLOR_BGR2GRAY);

        if (!initialized) {
            return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // Waiting for first GNSS anchor
        }

        if ((int)prev_kps_.size() < 30) {
            cv::goodFeaturesToTrack(curr_gray, prev_kps_, max_corners_, quality_level_, min_distance_, cv::Mat(), block_size_);
            if (!prev_kps_.empty()) {
                cv::cornerSubPix(curr_gray, prev_kps_, cv::Size(5, 5), cv::Size(-1, -1), subpix_criteria_);
            }
            prev_gray_   = curr_gray.clone();
            prev_depth_  = depth.clone();
            return get_pose_tuple();
        }

        // Forward Tracking
        std::vector<cv::Point2f> curr_kps;
        std::vector<uchar>       status_fwd;
        std::vector<float>       err;
        cv::calcOpticalFlowPyrLK(prev_gray_, curr_gray, prev_kps_, curr_kps, status_fwd, err, win_size_, max_level_, lk_criteria_);

        // Backward Tracking
        std::vector<cv::Point2f> kps_back;
        std::vector<uchar>       status_bwd;
        cv::calcOpticalFlowPyrLK(curr_gray, prev_gray_, curr_kps, kps_back, status_bwd, err, win_size_, max_level_, lk_criteria_);

        std::vector<cv::Point3f> obj_pts;
        std::vector<cv::Point2f> img_pts;
        std::vector<cv::Point2f> good_new;

        for (size_t i = 0; i < status_fwd.size(); ++i) {
            if (status_fwd[i] && status_bwd[i]) {
                double diff = cv::norm(prev_kps_[i] - kps_back[i]);
                if (diff < 1.0) { // Forward-Backward 1-pixel threshold
                    const cv::Point2f& old_pt = prev_kps_[i];
                    const cv::Point2f& new_pt = curr_kps[i];

                    int ix = static_cast<int>(old_pt.x);
                    int iy = static_cast<int>(old_pt.y);
                    
                    if (ix < 0 || ix >= width_ || iy < 0 || iy >= height_) continue;

                    float z = prev_depth_.at<float>(iy, ix);
                    if (z < 1.5f || z > 80.0f) continue;

                    float x3d = (old_pt.x - cx_) * z / fx_;
                    float y3d = (old_pt.y - cy_) * z / fy_;
                    
                    obj_pts.emplace_back(x3d, y3d, z);
                    img_pts.push_back(new_pt);
                    good_new.push_back(new_pt);
                }
            }
        }

        if ((int)obj_pts.size() >= 10) {
            cv::Mat rvec, tvec;
            bool success = cv::solvePnPRansac(
                obj_pts, img_pts, K_, dist_coeffs_,
                rvec, tvec, false, 150, 1.0f, 0.99, cv::noArray(), cv::SOLVEPNP_EPNP);

            if (success) {
                cv::Mat R_mat;
                cv::Rodrigues(rvec, R_mat);

                Eigen::Matrix3d R_eigen;
                Eigen::Vector3d t_eigen;
                for (int i = 0; i < 3; ++i) {
                    t_eigen(i) = tvec.at<double>(i);
                    for (int j = 0; j < 3; ++j) R_eigen(i,j) = R_mat.at<double>(i,j);
                }

                Eigen::Vector3d delta_t_cam = -R_eigen.transpose() * t_eigen;
                Eigen::Matrix3d R_rel_cam   =  R_eigen.transpose();

                Eigen::Vector3d delta_t_body = R_cam_to_body_ * delta_t_cam;
                Eigen::Matrix3d R_rel_body   = R_cam_to_body_ * R_rel_cam * R_cam_to_body_.transpose();

                // Motion Model Velocity Fusion
                double norm_vo = delta_t_body.norm();
                if (norm_vo > 1e-6) {
                    delta_t_body = (delta_t_body / norm_vo) * pred_dist;
                } else {
                    delta_t_body = Eigen::Vector3d(pred_dist, 0.0, 0.0);
                }

                Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
                Eigen::Vector3d delta_t_ego = delta_t_body + (I - R_rel_body) * lever_arm_;

                // Atomic Update
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    t_ego_world_ += R_ego_world_ * delta_t_ego;
                    R_ego_world_  = R_ego_world_ * R_rel_body;
                }
            }
        }

        prev_gray_  = curr_gray.clone();
        prev_depth_ = depth.clone();

        if ((int)good_new.size() < 30) {
            prev_kps_.clear(); // Force redetect next frame
        } else {
            prev_kps_ = good_new;
        }

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

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr          odom_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr       rgb_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr       depth_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       speed_sub_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr   gnss_front_sub_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr   gnss_rear_sub_;

    std::map<double, sensor_msgs::msg::Image::SharedPtr> rgb_cache_;
    std::mutex msg_mutex_;

    const double EARTH_RADIUS = 6378137.0;

public:
    VisualOdometryNode() : Node("visual_odometry_node"), vo_(90.0, 1280, 720) {
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/vo/odometry", 10);

        auto qos_sensor = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();

        rgb_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/carla/hero/front_left/image", qos_sensor,
            std::bind(&VisualOdometryNode::rgb_callback, this, std::placeholders::_1));

        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/inference/front_left/depth", qos_sensor,
            std::bind(&VisualOdometryNode::depth_callback, this, std::placeholders::_1));

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
        RCLCPP_INFO(this->get_logger(), "L4 Visual Odometry Node Active (Dual GNSS + Motion Model).");
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

    // Helper: Geodetic (Lat/Lon) to Local Cartesian (East, North, Up)
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
            // 🎯 FIX: Force standard CARLA map origin to match Ground Truth & EKF
            origin_lat_ = 0.0;
            origin_lon_ = 0.0;
            origin_alt_ = 0.0;
            has_origin_ = true;
        }

        Eigen::Vector3d f_loc = geo_to_enu(latest_gnss_front_->latitude, latest_gnss_front_->longitude, latest_gnss_front_->altitude);
        Eigen::Vector3d r_loc = geo_to_enu(latest_gnss_rear_->latitude, latest_gnss_rear_->longitude, latest_gnss_rear_->altitude);

        // Calculate Center ego position (GNSS Z is usually roof mounted, lower by 1.5m to reach base_link)
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
            if (best_key < 0) return;
            rgb_cache_.erase(best_key);
        }

        // 1Hz GNSS Anchoring
        if (frame_counter_ % 20 == 0) {
            process_gnss_snap();
        }

        // Wait until first GNSS ping sets origin before publishing anything
        if (!vo_.initialized) return;

        try {
            cv_bridge::CvImageConstPtr cv_rgb       = cv_bridge::toCvShare(best_rgb, "bgr8");
            cv_bridge::CvImageConstPtr cv_depth_ptr = cv_bridge::toCvShare(depth_msg, "32FC1");

            cv::Mat working_depth = cv_depth_ptr->image;
            if (working_depth.size() != cv_rgb->image.size()) {
                cv::resize(working_depth, working_depth, cv_rgb->image.size(), 0, 0, cv::INTER_NEAREST);
            }

            // Calculate DT and Predicted Distance for Motion Model
            rclcpp::Time current_time = depth_msg->header.stamp;
            double dt = (current_time - last_frame_time_).seconds();
            if (dt <= 0.0 || dt > 0.5) dt = 0.05; // Fallback for first frame or massive lag
            last_frame_time_ = current_time;
            
            double pred_dist = latest_speed_ * dt;

            std::vector<double> pose = vo_.process_frame(cv_rgb->image, working_depth, pred_dist);

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