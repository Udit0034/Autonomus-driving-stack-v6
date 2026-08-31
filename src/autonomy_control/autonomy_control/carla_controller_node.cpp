#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float32.hpp>
#include <carla_msgs/msg/carla_ego_vehicle_control.hpp>

#include <cmath>
#include <algorithm>
#include <optional>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

// ==========================================
// 1. PID Controller Class (Used for Speed)
// ==========================================
class PIDController {
public:
    double kp, ki, kd, output_min, output_max, integral_clamp;

    PIDController(double kp, double ki, double kd, 
                  double output_min = -1.0, double output_max = 1.0, 
                  double integral_clamp = 2.0)
        : kp(kp), ki(ki), kd(kd), output_min(output_min), 
          output_max(output_max), integral_clamp(integral_clamp),
          _integral(0.0), _prev_error(0.0), _first(true) {}

    double compute(double error, double dt) {
        if (dt <= 0.0) return 0.0;

        if (std::abs(error) > 0.25) {
            _integral += error * dt;
            _integral = std::clamp(_integral, -integral_clamp, integral_clamp);
        }

        double derivative = 0.0;
        if (_first) {
            _first = false;
        } else {
            derivative = (error - _prev_error) / dt;
        }
        _prev_error = error;

        double output = (kp * error) + (ki * _integral) + (kd * derivative);
        return std::clamp(output, output_min, output_max);
    }

private:
    double _integral;
    double _prev_error;
    bool _first;
};

// ==========================================
// 2. Longitudinal Controller (Unchanged)
// ==========================================
class LongitudinalController {
public:
    double max_accel, max_decel, max_jerk, max_jerk_emergency, velocity_filter_alpha;

    // PATCH: Adjusted kp and ki to overcome physics drag and reach 8.3 m/s
    LongitudinalController(double kp = 0.8, double ki = 0.05, double kd = 0.1, 
                           double max_accel = 2.5, double max_decel = 4.0,
                           double max_jerk = 2.5, double max_jerk_emergency = 5.0, 
                           double velocity_filter_alpha = 0.5)
        : max_accel(max_accel), max_decel(max_decel), max_jerk(max_jerk),
          max_jerk_emergency(max_jerk_emergency), velocity_filter_alpha(velocity_filter_alpha),
          _prev_accel(0.0), _filtered_speed(0.0), _emergency(false),
          _pid(kp, ki, kd, -max_decel, max_accel, 0.8) {}

    double compute(double current_speed, double target_speed, double dt) {
        double alpha = (current_speed < 2.0) ? 0.25 : velocity_filter_alpha;
        _filtered_speed = (alpha * current_speed) + ((1.0 - alpha) * _filtered_speed);
        _emergency = (target_speed < 0.1);

        if (_prev_target_speed.has_value()) {
            double rate = _emergency ? max_decel * 2.0 : 2.5;
            double max_change = rate * dt;
            target_speed = std::clamp(target_speed, 
                                      _prev_target_speed.value() - max_change, 
                                      _prev_target_speed.value() + max_change);
        }

        double a_ff = 0.0;
        if (_prev_target_speed.has_value() && dt > 0.0 && target_speed > 2.0) {
            a_ff = (target_speed - _prev_target_speed.value()) / dt;
        }
        a_ff = std::clamp(a_ff, -max_decel, max_accel);
        _prev_target_speed = target_speed;

        double error = target_speed - _filtered_speed;
        double accel_pid = _pid.compute(error, dt);
        double accel = std::clamp(a_ff + accel_pid, -max_decel, max_accel);

        if (dt > 0.0) {
            double jerk_limit = _emergency ? max_jerk_emergency : max_jerk;
            double max_change = jerk_limit * dt;
            accel = std::clamp(accel, _prev_accel - max_change, _prev_accel + max_change);
        }

        _prev_accel = accel;
        return accel;
    }

private:
    double _prev_accel;
    double _filtered_speed;
    std::optional<double> _prev_target_speed;
    bool _emergency;
    PIDController _pid;
};

class LateralController {
public:
    double wheelbase;
    double max_steer_rate;

    LateralController(double wheelbase = 2.87, double max_steer_rate = 2.5)
        : wheelbase(wheelbase), max_steer_rate(max_steer_rate), _prev_steer(0.0) {}

    double compute(const geometry_msgs::msg::Pose& target_pose,
                   const nav_msgs::msg::Odometry& current_odom,
                   double current_speed, double dt) {

        double ego_x = current_odom.pose.pose.position.x;
        double ego_y = current_odom.pose.pose.position.y;
        const auto& q = current_odom.pose.pose.orientation;
        double ego_yaw = std::atan2(2.0*(q.w*q.z + q.x*q.y),
                                    1.0 - 2.0*(q.y*q.y + q.z*q.z));

        // Transform global target into local ego frame (with RH flip)
        double dx    =  target_pose.position.x - ego_x;
        double dy_rh = -(target_pose.position.y - ego_y);
        double cos_y = std::cos(ego_yaw), sin_y = std::sin(ego_yaw);

        double lx = dx*cos_y + dy_rh*sin_y;
        double ly = -dx*sin_y + dy_rh*cos_y;

        double Ld    = std::max(std::hypot(lx, ly), 1.0);
        double alpha = std::atan2(ly, lx);
        double steer_rad = std::atan((2.0*wheelbase*std::sin(alpha)) / Ld);
        double steer_cmd = std::clamp(-steer_rad / 1.22, -1.0, 1.0);

        if (dt > 0.0) {
            double mc = max_steer_rate * dt;
            steer_cmd = std::clamp(steer_cmd, _prev_steer - mc, _prev_steer + mc);
        }
        _prev_steer = steer_cmd;
        return steer_cmd;
    }

private:
    double _prev_steer;
};

// ==========================================
// 4. ROS 2 Node 
// ==========================================
class CarlaControllerNode : public rclcpp::Node {
public:
    CarlaControllerNode() : Node("carla_controller_node") {
        rclcpp::QoS sensor_qos = rclcpp::SensorDataQoS();
        rclcpp::QoS reliable_qos = rclcpp::SystemDefaultsQoS().reliable().keep_last(10);

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ekf/odometry", sensor_qos,
            std::bind(&CarlaControllerNode::odom_callback, this, std::placeholders::_1));

        target_speed_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "/planning/target_speed", reliable_qos,
            std::bind(&CarlaControllerNode::target_speed_callback, this, std::placeholders::_1));

        target_waypoint_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/planning/target_waypoint", reliable_qos,
            std::bind(&CarlaControllerNode::target_waypoint_callback, this, std::placeholders::_1));

        control_pub_ = this->create_publisher<carla_msgs::msg::CarlaEgoVehicleControl>(
            "/carla/hero/vehicle_control_cmd", reliable_qos);

        last_time_ = this->now();
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&CarlaControllerNode::control_loop, this));

        RCLCPP_INFO(this->get_logger(), "🏎️ V7 Carla Controller Online: Running Pure Pursuit @ 50Hz.");
    }

private:
    LongitudinalController lon_controller_;
    LateralController lat_controller_;

    nav_msgs::msg::Odometry latest_odom_;
    double current_speed_ = 0.0;
    double target_speed_ = 0.0;
    geometry_msgs::msg::Pose target_pose_;
    
    bool has_odom_ = false;
    bool has_target_wp_ = false;

    rclcpp::Time last_time_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr target_speed_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_waypoint_sub_;
    rclcpp::Publisher<carla_msgs::msg::CarlaEgoVehicleControl>::SharedPtr control_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        latest_odom_ = *msg;
        current_speed_ = std::sqrt(std::pow(msg->twist.twist.linear.x, 2) + 
                                   std::pow(msg->twist.twist.linear.y, 2));
        has_odom_ = true;
    }

    void target_speed_callback(const std_msgs::msg::Float32::SharedPtr msg) {
        target_speed_ = msg->data;
    }

    void target_waypoint_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        // Now expecting a GLOBAL map-frame waypoint
        target_pose_ = msg->pose;
        has_target_wp_ = true;
    }

    void control_loop() {
        if (!has_odom_ || !has_target_wp_) {
            return; 
        }

        rclcpp::Time current_time = this->now();
        double dt = (current_time - last_time_).seconds();
        last_time_ = current_time;

        if (dt <= 0.0) return;

        // 1. Calculate Commands
        double accel_cmd = lon_controller_.compute(current_speed_, target_speed_, dt);
        
        // PASS THE FRESH ODOMETRY DIRECTLY INTO THE LATERAL CONTROLLER
        // PASS CURRENT SPEED TO STANLEY CONTROLLER
        double steer_cmd = lat_controller_.compute(target_pose_, latest_odom_, current_speed_, dt);

        // 2. Map to CARLA Throttle/Brake Envelopes
        carla_msgs::msg::CarlaEgoVehicleControl drive_msg;
        drive_msg.header.stamp = current_time;
        drive_msg.steer = steer_cmd;
        
        if (accel_cmd >= 0.0) {
            drive_msg.throttle = std::min(1.0, accel_cmd / lon_controller_.max_accel);
            drive_msg.brake = 0.0;
        } else {
            drive_msg.throttle = 0.0;
            drive_msg.brake = std::min(1.0, std::abs(accel_cmd) / lon_controller_.max_decel);
        }

        if (target_speed_ < 0.1 && current_speed_ < 0.5) {
            drive_msg.hand_brake = true;
        } else {
            drive_msg.hand_brake = false;
        }

        control_pub_->publish(drive_msg);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CarlaControllerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}