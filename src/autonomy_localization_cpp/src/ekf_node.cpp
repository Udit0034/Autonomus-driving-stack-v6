#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

// Message Filters for Dual GNSS Sync
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

// Eigen for Matrix Math
#include <Eigen/Dense>

// TF2 Headers for Coordinate Transforms
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <algorithm> 
#include <cmath>
#include <vector>
#include <tuple>
#include <memory>

using std::placeholders::_1;
using std::placeholders::_2;

// ==========================================
// MATH UTILITIES (Powered by TF2)
// ==========================================
double wrap_angle(double angle) {
    return std::remainder(angle, 2.0 * M_PI);
}

Eigen::Vector4d normalize_quaternion(const Eigen::Vector4d& q) {
    double n = q.norm();
    if (n < 1e-12) return Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    return q / n;
}

Eigen::Vector4d quaternion_multiply(const Eigen::Vector4d& q1, const Eigen::Vector4d& q2) {
    double w1 = q1(0), x1 = q1(1), y1 = q1(2), z1 = q1(3);
    double w2 = q2(0), x2 = q2(1), y2 = q2(2), z2 = q2(3);
    return Eigen::Vector4d(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    );
}

Eigen::Vector4d small_angle_quaternion(const Eigen::Vector3d& delta_theta) {
    double theta = delta_theta.norm();
    if (theta < 1e-10) {
        return normalize_quaternion(Eigen::Vector4d(1.0, 0.5 * delta_theta(0), 0.5 * delta_theta(1), 0.5 * delta_theta(2)));
    }
    Eigen::Vector3d axis = delta_theta / theta;
    double half = 0.5 * theta;
    Eigen::Vector4d q;
    q(0) = std::cos(half);
    q.segment<3>(1) = axis * std::sin(half);
    return q;
}

Eigen::Vector4d euler_to_quaternion(double yaw, double pitch, double roll) {
    tf2::Quaternion q;
    q.setRPY(roll, pitch, yaw); // TF2 uses Roll, Pitch, Yaw order
    return normalize_quaternion(Eigen::Vector4d(q.w(), q.x(), q.y(), q.z()));
}

std::tuple<double, double, double> quaternion_to_euler(const Eigen::Vector4d& q_in) {
    Eigen::Vector4d norm_q = normalize_quaternion(q_in);
    tf2::Quaternion q(norm_q(1), norm_q(2), norm_q(3), norm_q(0)); // TF2 constructor is x, y, z, w
    
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    
    return {wrap_angle(yaw), wrap_angle(pitch), wrap_angle(roll)};
}

double quaternion_to_yaw(const Eigen::Vector4d& q) {
    auto [yaw, pitch, roll] = quaternion_to_euler(q);
    return yaw;
}

// ==========================================
// EKF CLASS
// ==========================================
class ESEKF12 {
public:
    static constexpr int IX = 0, IY = 1, IZ = 2;
    static constexpr int IQW = 3, IQX = 4, IQY = 5, IQZ = 6;
    static constexpr int IVX = 7, IVY = 8, IVZ = 9;
    static constexpr int IBGX = 10, IBGY = 11, IBGZ = 12;

    static constexpr int NOMINAL_DIM = 13;
    static constexpr int ERROR_DIM = 12;

    Eigen::Matrix<double, NOMINAL_DIM, 1> x;
    Eigen::Matrix<double, ERROR_DIM, ERROR_DIM> P;
    Eigen::Matrix<double, ERROR_DIM, ERROR_DIM> Q;

    Eigen::Matrix3d R_gnss;
    Eigen::Matrix3d R_vel_3d;
    Eigen::Matrix<double, 1, 1> R_compass, R_dual_gnss_yaw, R_gnss_yaw, R_altimeter, R_wheel_odom, R_slam_yaw;
    Eigen::Matrix2d R_pitch_roll;
    Eigen::Matrix3d R_vo_pos, R_vo_orient, R_slam_pos;

    bool initialized_ = false;

    ESEKF12() {
        x.setZero();
        x(IQW) = 1.0;
        P = Eigen::Matrix<double, ERROR_DIM, ERROR_DIM>::Identity() * 1.0;

        Eigen::Matrix<double, ERROR_DIM, 1> q_diag;
        // Process Noise Covariance (Q) - Mapped from q_dp, q_dtheta, q_dv, q_dbg
        q_diag << 0.012108313207417919, 0.02510667237511935, 8.417594573577902e-08, 
                  0.0030121773475646923, 0.0007164572639373572, 0.11319064167811974, 
                  1.2934951030921336e-06, 0.009333958142352715, 0.0006649396931700656, 
                  1.4890217204619358e-06, 0.00015295583702522328, 0.0016202984415564913;
        Q = q_diag.asDiagonal();

        // Measurement Noise Covariance (R)
        R_gnss = Eigen::Vector3d(0.013914231175931305, 0.08937575706652634, 0.8156559073177112).asDiagonal();
        R_vel_3d = Eigen::Vector3d(0.016213595883410377, 0.023221647088025914, 0.03697263125409558).asDiagonal();
        
        R_compass << 44.07732232405785;
        R_dual_gnss_yaw << 0.03829683444774799;
        R_gnss_yaw << 0.17031386929923029;
        R_pitch_roll = Eigen::Vector2d(0.030414472460481096, 0.11428522568956152).asDiagonal();
        
        R_vo_pos = Eigen::Vector3d(6.286180939125132, 0.32167055836732544, 0.022849847460362137).asDiagonal();
        R_vo_orient = Eigen::Vector3d(0.028488773650616295, 0.06239567836653208, 5.0086767076152245).asDiagonal();
        
        R_altimeter << 12.912056797548939; 
        R_wheel_odom << 20.696165418399502;

        R_slam_pos = Eigen::Vector3d(40.315724178492985, 45.05090729338225, 42.7440272512942).asDiagonal();
        R_slam_yaw << 98.71277256764778;
    }

    void initialize_state(double x_val, double y_val, double z_val, double vx, double yaw, double pitch, double roll, 
                          double vy = 0.0, double vz = 0.0, double bias_x = 0.0, double bias_y = 0.0, double bias_z = 0.0) 
    {
        Eigen::Vector4d q = euler_to_quaternion(yaw, pitch, roll);
        x << x_val, y_val, z_val, q(0), q(1), q(2), q(3), vx, vy, vz, bias_x, bias_y, bias_z;
        initialized_ = true;
    }

    Eigen::Vector4d get_quaternion() const { return x.segment<4>(IQW); }
    
    std::tuple<double, double, double> get_euler() const { return quaternion_to_euler(get_quaternion()); }

    template<int MeasDim>
    void error_update(const Eigen::Matrix<double, MeasDim, 1>& residual,
                      const Eigen::Matrix<double, MeasDim, ERROR_DIM>& H,
                      const Eigen::Matrix<double, MeasDim, MeasDim>& R,
                      const std::vector<int>& angle_rows = {}) 
    {
        Eigen::Matrix<double, MeasDim, 1> y = residual;
        for (int idx : angle_rows) y(idx) = wrap_angle(y(idx));

        Eigen::Matrix<double, MeasDim, MeasDim> S = H * P * H.transpose() + R;
        Eigen::Matrix<double, ERROR_DIM, MeasDim> K = P * H.transpose() * S.inverse();

        Eigen::Matrix<double, ERROR_DIM, 1> delta_x = K * y;
        inject_error_state(delta_x);

        Eigen::Matrix<double, ERROR_DIM, ERROR_DIM> I_KH = Eigen::Matrix<double, ERROR_DIM, ERROR_DIM>::Identity() - K * H;
        P = I_KH * P * I_KH.transpose() + K * R * K.transpose();
    }

    void predict(double dt, double accel_x, double accel_y, double gyro_x, double gyro_y, double gyro_z) {
        if (!initialized_ || dt <= 0.0) return;

        Eigen::Vector3d p = x.segment<3>(IX);
        Eigen::Vector4d q = get_quaternion();
        Eigen::Vector3d v = x.segment<3>(IVX);
        Eigen::Vector3d bg = x.segment<3>(IBGX);

        Eigen::Vector3d omega(gyro_x, gyro_y, gyro_z);
        omega -= bg;

        Eigen::Vector4d dq = small_angle_quaternion(omega * dt);
        Eigen::Vector4d q_new = normalize_quaternion(quaternion_multiply(q, dq));
        double yaw = quaternion_to_yaw(q_new);

        Eigen::Vector3d a_world(
            accel_x * std::cos(yaw) - accel_y * std::sin(yaw),
            accel_x * std::sin(yaw) + accel_y * std::cos(yaw),
            0.0
        );

        Eigen::Vector3d v_new = v + a_world * dt;
        Eigen::Vector3d p_new = p + v * dt;

        x.segment<3>(IX) = p_new;
        x.segment<4>(IQW) = q_new;
        x.segment<3>(IVX) = v_new;

        Eigen::Matrix<double, ERROR_DIM, ERROR_DIM> F = Eigen::Matrix<double, ERROR_DIM, ERROR_DIM>::Identity();
        F.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
        F.block<3, 3>(3, 9) = -Eigen::Matrix3d::Identity() * dt;
        F(6, 5) = (-accel_x * std::sin(yaw) - accel_y * std::cos(yaw)) * dt;
        F(7, 5) = ( accel_x * std::cos(yaw) - accel_y * std::sin(yaw)) * dt;

        Eigen::Matrix<double, ERROR_DIM, ERROR_DIM> q_dt = Q * std::max(dt, 1e-3);
        P = F * P * F.transpose() + q_dt;
    }

    void update_gnss_3d(double meas_x, double meas_y, double meas_z) {
        if (!initialized_) return;
        Eigen::Vector3d z(meas_x, meas_y, meas_z);
        Eigen::Vector3d residual = z - x.segment<3>(IX);
        Eigen::Matrix<double, 3, ERROR_DIM> H = Eigen::Matrix<double, 3, ERROR_DIM>::Zero();
        H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        error_update<3>(residual, H, R_gnss);
    }

    void update_compass_yaw(double meas_yaw) {
        if (!initialized_) return;
        auto [yaw, pitch, roll] = get_euler();
        Eigen::Matrix<double, 1, 1> residual; residual << wrap_angle(meas_yaw - yaw);
        Eigen::Matrix<double, 1, ERROR_DIM> H = Eigen::Matrix<double, 1, ERROR_DIM>::Zero();
        H(0, 5) = 1.0;
        error_update<1>(residual, H, R_compass, {0});
    }

    void update_gnss_yaw(double meas_yaw) {
        if (!initialized_) return;
        auto [yaw, pitch, roll] = get_euler();
        Eigen::Matrix<double, 1, 1> residual; residual << wrap_angle(meas_yaw - yaw);
        Eigen::Matrix<double, 1, ERROR_DIM> H = Eigen::Matrix<double, 1, ERROR_DIM>::Zero();
        H(0, 5) = 1.0;
        error_update<1>(residual, H, R_gnss_yaw, {0});
    }

    void update_dual_gnss_yaw(double meas_yaw) {
        if (!initialized_) return;
        auto [yaw, pitch, roll] = get_euler();
        Eigen::Matrix<double, 1, 1> residual; residual << wrap_angle(meas_yaw - yaw);
        Eigen::Matrix<double, 1, ERROR_DIM> H = Eigen::Matrix<double, 1, ERROR_DIM>::Zero();
        H(0, 5) = 1.0;
        error_update<1>(residual, H, R_dual_gnss_yaw, {0});
    }

    void update_pitch_roll_from_accel(double meas_pitch, double meas_roll, double accel_magnitude) {
        if (!initialized_) return;
        if (std::abs(accel_magnitude - 9.81) > 0.5 && accel_magnitude > 0.5) return;
        auto [yaw, pitch, roll] = get_euler();
        Eigen::Vector2d residual(meas_pitch - pitch, meas_roll - roll);
        Eigen::Matrix<double, 2, ERROR_DIM> H = Eigen::Matrix<double, 2, ERROR_DIM>::Zero();
        H(0, 4) = 1.0; 
        H(1, 3) = 1.0; 
        error_update<2>(residual, H, R_pitch_roll, {0, 1});
    }

    void update_altimeter(double meas_z) {
        if (!initialized_) return;
        Eigen::Matrix<double, 1, 1> residual; residual << meas_z - x(IZ);
        Eigen::Matrix<double, 1, ERROR_DIM> H = Eigen::Matrix<double, 1, ERROR_DIM>::Zero();
        H(0, 2) = 1.0;
        error_update<1>(residual, H, R_altimeter);
    }

    void update_wheel_odom(double meas_speed) {
        if (!initialized_) return;
        double yaw = quaternion_to_yaw(get_quaternion());
        double v_forward_est = x(IVX) * std::cos(yaw) + x(IVY) * std::sin(yaw);
        Eigen::Matrix<double, 1, 1> residual; residual << meas_speed - v_forward_est;
        Eigen::Matrix<double, 1, ERROR_DIM> H = Eigen::Matrix<double, 1, ERROR_DIM>::Zero();
        H(0, 6) = std::cos(yaw);
        H(0, 7) = std::sin(yaw);
        error_update<1>(residual, H, R_wheel_odom);
    }

    void update_vo_position(double meas_x, double meas_y, double meas_z) {
        if (!initialized_) return;
        Eigen::Vector3d residual(meas_x - x(IX), meas_y - x(IY), meas_z - x(IZ));
        Eigen::Matrix<double, 3, ERROR_DIM> H = Eigen::Matrix<double, 3, ERROR_DIM>::Zero();
        H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        error_update<3>(residual, H, R_vo_pos);
    }

    void update_slam_pose(double meas_x, double meas_y, double meas_z) {
        if (!initialized_) return;
        Eigen::Vector3d residual(meas_x - x(IX), meas_y - x(IY), meas_z - x(IZ));
        Eigen::Matrix<double, 3, ERROR_DIM> H = Eigen::Matrix<double, 3, ERROR_DIM>::Zero();
        H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        error_update<3>(residual, H, R_slam_pos);
    }

    void update_slam_yaw(double meas_yaw) {
        if (!initialized_) return;
        auto [yaw, pitch, roll] = get_euler();
        Eigen::Matrix<double, 1, 1> residual; residual << wrap_angle(meas_yaw - yaw);
        Eigen::Matrix<double, 1, ERROR_DIM> H = Eigen::Matrix<double, 1, ERROR_DIM>::Zero();
        H(0, 5) = 1.0;
        error_update<1>(residual, H, R_slam_yaw, {0});
    }

private:
    void inject_error_state(const Eigen::Matrix<double, ERROR_DIM, 1>& delta_x) {
        x.segment<3>(IX) += delta_x.segment<3>(0);
        Eigen::Vector4d q = get_quaternion();
        Eigen::Vector4d dq = small_angle_quaternion(delta_x.segment<3>(3));
        x.segment<4>(IQW) = normalize_quaternion(quaternion_multiply(q, dq));
        x.segment<3>(IVX) += delta_x.segment<3>(6);
        x.segment<3>(IBGX) += delta_x.segment<3>(9);
    }
};

// ==========================================
// ROS2 NODE INTEGRATION
// ==========================================
class EKFFusionNode : public rclcpp::Node {
public:
    EKFFusionNode() : Node("ekf_fusion_node") {
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/ekf/odometry", 10);
        
        // Initialize the TF2 Broadcaster
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/carla/hero/imu", 10, std::bind(&EKFFusionNode::imu_callback, this, _1));
        vo_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/vo/odometry", 10, std::bind(&EKFFusionNode::vo_callback, this, _1));
        slam_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/slam/odometry", 10, std::bind(&EKFFusionNode::slam_callback, this, _1));
        altimeter_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
            "/carla/altimeter", 10, std::bind(&EKFFusionNode::altimeter_callback, this, _1));
        wheel_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/carla/hero/wheel_odom", 10, std::bind(&EKFFusionNode::wheel_odom_callback, this, _1));

        // Setup Dual GNSS Message Filters
        gnss_front_sub_.subscribe(this, "/carla/hero/gnss_front");
        gnss_rear_sub_.subscribe(this, "/carla/hero/gnss_rear");
        
        gnss_sync_ = std::make_shared<message_filters::Synchronizer<ApproxSyncPolicy>>(
            ApproxSyncPolicy(10), gnss_front_sub_, gnss_rear_sub_);
        gnss_sync_->registerCallback(std::bind(&EKFFusionNode::dual_gnss_callback, this, _1, _2));

        RCLCPP_INFO(this->get_logger(), "EKF Fusion Node Started. Waiting for GNSS/IMU to initialize...");
    }

private:
    ESEKF12 ekf_;

    double last_time_ = -1.0;
    std::vector<double> gnss_origin_ = {0.0, 0.0, 0.0};
    
    bool has_prev_gnss_ = false;
    double prev_gnss_x_ = 0.0, prev_gnss_y_ = 0.0, prev_gnss_time_ = 0.0;

    // Publishers & Broadcasters
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // Standard Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr vo_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr slam_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr altimeter_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_odom_sub_;
    
    // Message Filters
    using NavSatFixMsg = sensor_msgs::msg::NavSatFix;
    message_filters::Subscriber<NavSatFixMsg> gnss_front_sub_;
    message_filters::Subscriber<NavSatFixMsg> gnss_rear_sub_;
    using ApproxSyncPolicy = message_filters::sync_policies::ApproximateTime<NavSatFixMsg, NavSatFixMsg>;
    std::shared_ptr<message_filters::Synchronizer<ApproxSyncPolicy>> gnss_sync_;

    double get_time_sec(const rclcpp::Time& stamp) {
        return stamp.seconds();
    }

    std::pair<double, double> latlon_to_xy(double lat, double lon) {
        const double EARTH_RADIUS = 6378137.0;
        double lat0 = gnss_origin_[0];
        double lon0 = gnss_origin_[1];
        
        double dlat = (lat - lat0) * M_PI / 180.0;
        double dlon = (lon - lon0) * M_PI / 180.0;
        
        double x = dlon * std::cos(lat0 * M_PI / 180.0) * EARTH_RADIUS;
        double y = dlat * EARTH_RADIUS;
        return {x, y};
    }

    void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr msg) {
        double current_time = get_time_sec(msg->header.stamp);
        
        if (last_time_ < 0) {
            last_time_ = current_time;
            return;
        }
            
        double dt = std::min(current_time - last_time_, 0.1);
        last_time_ = current_time;

        if (!ekf_.initialized_) return;

        double accel_x = msg->linear_acceleration.x;
        double accel_y = msg->linear_acceleration.y;
        double accel_z = msg->linear_acceleration.z;
        double gyro_x = msg->angular_velocity.x;
        double gyro_y = msg->angular_velocity.y;
        double gyro_z = msg->angular_velocity.z;

        // Note: Python implementation uses inverted Y for IMU prop
        ekf_.predict(dt, accel_x, accel_y, gyro_x, gyro_y, gyro_z);

        Eigen::Vector4d q(msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z);
        auto [yaw, pitch, roll] = quaternion_to_euler(q);
        double compass_yaw = wrap_angle(yaw + M_PI / 2.0);

        ekf_.update_compass_yaw(compass_yaw);
        
        double accel_mag = std::sqrt(accel_x*accel_x + accel_y*accel_y + accel_z*accel_z);
        ekf_.update_pitch_roll_from_accel(pitch, roll, accel_mag);

        publish_ekf_state(this->now());
    }

    void dual_gnss_callback(const NavSatFixMsg::ConstSharedPtr front_msg, const NavSatFixMsg::ConstSharedPtr rear_msg) {
        double current_time = get_time_sec(front_msg->header.stamp);

        auto [curr_front_x, curr_front_y] = latlon_to_xy(front_msg->latitude, front_msg->longitude);
        auto [curr_rear_x, curr_rear_y] = latlon_to_xy(rear_msg->latitude, rear_msg->longitude);
        double curr_front_z = front_msg->altitude - gnss_origin_[2];

        double dx = curr_front_x - curr_rear_x;
        double dy = curr_front_y - curr_rear_y;
        double dual_gnss_yaw = std::atan2(dy, dx);

        // --- BUG 2 FIX: Lever arm correction ---
        // Project the position back to the vehicle center
        constexpr double FRONT_ANTENNA_OFFSET = 1.11;  // meters forward from center
        double vehicle_x = curr_front_x - FRONT_ANTENNA_OFFSET * std::cos(dual_gnss_yaw);
        double vehicle_y = curr_front_y - FRONT_ANTENNA_OFFSET * std::sin(dual_gnss_yaw);

        if (!ekf_.initialized_) {
            ekf_.initialize_state(vehicle_x, vehicle_y, curr_front_z, 0.0, dual_gnss_yaw, 0.0, 0.0);
            prev_gnss_time_ = current_time;
            prev_gnss_x_ = vehicle_x;
            prev_gnss_y_ = vehicle_y;
            has_prev_gnss_ = true;
            RCLCPP_INFO(this->get_logger(), "EKF Initialized at X:%.2f, Y:%.2f | Yaw: %.2f°", 
                        vehicle_x, vehicle_y, dual_gnss_yaw * 180.0 / M_PI);
            return;
        }

        // Use vehicle center coordinates for all EKF updates
        ekf_.update_gnss_3d(vehicle_x, vehicle_y, curr_front_z);

        if (has_prev_gnss_) {
            double gnss_dt = current_time - prev_gnss_time_;
            if (gnss_dt > 0.0) {
                double vx = (vehicle_x - prev_gnss_x_) / gnss_dt;
                double vy = (vehicle_y - prev_gnss_y_) / gnss_dt;
                double speed = std::sqrt(vx*vx + vy*vy);
                if (speed > 1.0) {
                    double gnss_yaw = std::atan2(vy, vx);
                    ekf_.update_gnss_yaw(gnss_yaw);
                }
            }
        }

        ekf_.update_dual_gnss_yaw(dual_gnss_yaw);

        prev_gnss_x_ = vehicle_x;
        prev_gnss_y_ = vehicle_y;
        prev_gnss_time_ = current_time;
    }

    void altimeter_callback(const geometry_msgs::msg::PointStamped::ConstSharedPtr msg) {
        ekf_.update_altimeter(msg->point.z);
    }

    void wheel_odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
        ekf_.update_wheel_odom(msg->twist.twist.linear.x);
    }

    void vo_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
        if (!ekf_.initialized_) return;

        ekf_.update_vo_position(
            msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            msg->pose.pose.position.z
        );
    }

    void slam_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
        if (!ekf_.initialized_) return;

        double slam_x = msg->pose.pose.position.x;
        double slam_y = msg->pose.pose.position.y;
        double slam_z = msg->pose.pose.position.z;

        Eigen::Vector4d q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, 
                          msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
        auto [slam_yaw, slam_pitch, slam_roll] = quaternion_to_euler(q);

        ekf_.update_slam_pose(slam_x, slam_y, slam_z);
        ekf_.update_slam_yaw(slam_yaw);
    }

    void publish_ekf_state(const rclcpp::Time& stamp) {
        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header.stamp = stamp;
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "base_link";

        odom_msg.pose.pose.position.x = ekf_.x(ESEKF12::IX);
        odom_msg.pose.pose.position.y = ekf_.x(ESEKF12::IY);
        odom_msg.pose.pose.position.z = ekf_.x(ESEKF12::IZ);

        odom_msg.twist.twist.linear.x = ekf_.x(ESEKF12::IVX);
        odom_msg.twist.twist.linear.y = ekf_.x(ESEKF12::IVY);
        odom_msg.twist.twist.linear.z = ekf_.x(ESEKF12::IVZ);

        Eigen::Vector4d q = ekf_.get_quaternion();
        odom_msg.pose.pose.orientation.w = q(0);
        odom_msg.pose.pose.orientation.x = -q(1);
        odom_msg.pose.pose.orientation.y = q(2);
        odom_msg.pose.pose.orientation.z = -q(3);

        odom_msg.twist.twist.angular.x = ekf_.x(ESEKF12::IBGX);
        odom_msg.twist.twist.angular.y = ekf_.x(ESEKF12::IBGY);
        odom_msg.twist.twist.angular.z = ekf_.x(ESEKF12::IBGZ);

        odom_pub_->publish(odom_msg);

        // Broadcast the TF2 Transform
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = stamp;
        t.header.frame_id = "odom";
        t.child_frame_id = "base_link";

        t.transform.translation.x = ekf_.x(ESEKF12::IX);
        t.transform.translation.y = ekf_.x(ESEKF12::IY);
        t.transform.translation.z = ekf_.x(ESEKF12::IZ);

        t.transform.rotation.w = q(0);
        t.transform.rotation.x = -q(1);
        t.transform.rotation.y = q(2);
        t.transform.rotation.z = -q(3);

        tf_broadcaster_->sendTransform(t);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EKFFusionNode>();
    
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    
    executor.spin();
    rclcpp::shutdown();
    return 0;
}