#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp> 

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <mutex>
#include <algorithm>
#include <fstream>
#include <limits>
#include <iomanip>

class CubicSpline {
public:
    CubicSpline() = default;
    CubicSpline(const std::vector<float>& x, const std::vector<float>& y) {
        int n = x.size();
        if (n < 3) return; 
        
        x_ = x; y_ = y; a_ = y;
        b_.resize(n, 0.0f); c_.resize(n, 0.0f); d_.resize(n, 0.0f);

        std::vector<float> h(n - 1);
        for (int i = 0; i < n - 1; ++i) h[i] = x[i + 1] - x[i];

        Eigen::MatrixXf A = Eigen::MatrixXf::Zero(n, n);
        Eigen::VectorXf B = Eigen::VectorXf::Zero(n);

        A(0, 0) = 1.0f; A(n - 1, n - 1) = 1.0f;

        for (int i = 1; i < n - 1; ++i) {
            A(i, i - 1) = h[i - 1];
            A(i, i) = 2.0f * (h[i - 1] + h[i]);
            A(i, i + 1) = h[i];
            B(i) = 3.0f * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
        }

        Eigen::VectorXf c_eigen = A.colPivHouseholderQr().solve(B);
        for (int i = 0; i < n; ++i) c_[i] = c_eigen(i);

        for (int i = 0; i < n - 1; ++i) {
            b_[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c_[i + 1] + 2.0f * c_[i]) / 3.0f;
            d_[i] = (c_[i + 1] - c_[i]) / (3.0f * h[i]);
        }
        valid_ = true;
    }

    float calc(float t) const {
        if (!valid_) return 0.0f;
        auto it = std::lower_bound(x_.begin(), x_.end(), t);
        int i = std::distance(x_.begin(), it) - 1;
        i = std::clamp(i, 0, static_cast<int>(x_.size()) - 2);
        float dx = t - x_[i];
        return a_[i] + b_[i] * dx + c_[i] * dx * dx + d_[i] * dx * dx * dx;
    }

    float calc_deriv(float t) const {
        if (!valid_) return 0.0f;
        auto it = std::lower_bound(x_.begin(), x_.end(), t);
        int i = std::distance(x_.begin(), it) - 1;
        i = std::clamp(i, 0, static_cast<int>(x_.size()) - 2);
        float dx = t - x_[i];
        return b_[i] + 2.0f * c_[i] * dx + 3.0f * d_[i] * dx * dx;
    }

    bool is_valid() const { return valid_; }

private:
    std::vector<float> x_, y_, a_, b_, c_, d_;
    bool valid_ = false;
};

class LatticePlannerNode : public rclcpp::Node {
public:
    LatticePlannerNode() : Node("lattice_planner_node") {
        lookahead_dist_ = 25.0f;
        lattice_deltas_ = {0.0f, -0.75f, 0.75f, -1.5f, 1.5f, -3.0f, 3.0f};
        lane_offset_abs_limit_ = 4.5f;
        last_chosen_offset_ = 0.0f;

        target_dist_ = 12.0f;
        map_res_ = 0.25f;
        map_center_ = 120;
        target_lane_offset_ = 0.0f;
        has_odom_ = false; has_costmap_ = false; has_semantic_ = false;
        
        last_closest_idx_ = 0; 
        was_turning_ = false;  

        gt_log_.open("/tmp/gt_trajectory_log.csv");
        gt_log_ << "t_sec,x,y,yaw\n";
        lat_log_.open("/tmp/lattice_planner_log.csv");
        
        lat_log_ << "t_sec,ego_x,ego_y,ego_yaw,tgt_offset,chosen_offset,chosen_cost,"
                 << "blocked_count,straight_blocked,wp_x,wp_y,target_dist,min_path_dist,ref_sane\n";

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ekf/odometry", 10, std::bind(&LatticePlannerNode::odom_callback, this, std::placeholders::_1));
        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/slam/local_costmap", 10, std::bind(&LatticePlannerNode::costmap_callback, this, std::placeholders::_1));
        semantic_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/perception/semantic_costmap", 10, std::bind(&LatticePlannerNode::semantic_callback, this, std::placeholders::_1));
        lane_offset_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "/behavior/target_lane_offset", 10, std::bind(&LatticePlannerNode::lane_offset_callback, this, std::placeholders::_1));
        global_path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/planning/global_path", 10, std::bind(&LatticePlannerNode::global_path_callback, this, std::placeholders::_1));

        wp_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/planning/target_waypoint", 10);
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/planning/local_path", 10);
        
        hazard_pub_ = this->create_publisher<std_msgs::msg::Bool>("/planning/hazard_detected", 10);

        timer_ = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&LatticePlannerNode::plan_local_path, this));
        RCLCPP_INFO(this->get_logger(), "🕸️ True Frenet-Frame Lattice Planner (Dynamic Cornering + LH/RH Fix) Online.");
    }

private:
    float lookahead_dist_, target_dist_, map_res_, target_lane_offset_;
    int map_center_;
    std::vector<float> lattice_deltas_;
    float lane_offset_abs_limit_, last_chosen_offset_;

    std::mutex data_mutex_;
    nav_msgs::msg::Odometry latest_odom_;
    bool has_odom_, has_costmap_, has_semantic_;
    nav_msgs::msg::OccupancyGrid latest_costmap_, latest_semantic_;
    float sem_res_;
    int sem_cx_, sem_cy_;
    std::vector<std::pair<float, float>> global_path_;
    
    size_t last_closest_idx_; 
    bool was_turning_; 

    std::ofstream gt_log_, lat_log_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr semantic_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr lane_offset_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr global_path_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr wp_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr hazard_pub_; 
    rclcpp::TimerBase::SharedPtr timer_;

    static inline int round_idx(float v, float res) { return static_cast<int>(std::floor(v / res + 0.5f)); }

    float get_ego_yaw(const nav_msgs::msg::Odometry& odom) {
        const auto& q = odom.pose.pose.orientation;
        return std::atan2(2.0f * (q.w * q.z + q.x * q.y), 1.0f - 2.0f * (q.y * q.y + q.z * q.z));
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        latest_odom_ = *msg;
        has_odom_ = true;
        
        gt_log_ << std::fixed << std::setprecision(3)
                << this->get_clock()->now().seconds() << ","
                << msg->pose.pose.position.x << "," << msg->pose.pose.position.y << ","
                << get_ego_yaw(*msg) << "\n";
    }

    void costmap_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        latest_costmap_ = *msg; map_res_ = msg->info.resolution; map_center_ = msg->info.height / 2;
        has_costmap_ = true;
    }

    void semantic_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        latest_semantic_ = *msg; sem_res_ = msg->info.resolution;
        sem_cx_ = static_cast<int>(std::round(-msg->info.origin.position.x / sem_res_));
        sem_cy_ = static_cast<int>(std::round(-msg->info.origin.position.y / sem_res_));
        has_semantic_ = true;
    }

    void lane_offset_callback(const std_msgs::msg::Float32::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        target_lane_offset_ = -msg->data;
    }

    void global_path_callback(const nav_msgs::msg::Path::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        bool path_changed = (msg->poses.size() != global_path_.size());
        global_path_.clear();
        for (const auto& pose : msg->poses) {
            global_path_.emplace_back(pose.pose.position.x, pose.pose.position.y);
        }

        if (path_changed) {
            last_closest_idx_ = 0;
            last_chosen_offset_ = 0.0f;
            was_turning_ = false;
        }
    }

    bool is_hard_blocked(const Eigen::ArrayXf& path_x, const Eigen::ArrayXf& path_y) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (!has_costmap_) return false;
        
        int w = latest_costmap_.info.width, h = latest_costmap_.info.height;
        int hit_count = 0;
        
        for (int i = 0; i < path_x.size(); ++i) {
            if (path_x[i] < 2.0f) continue; 
            
            int col = map_center_ - round_idx(path_y[i], map_res_);
            int row = map_center_ - round_idx(path_x[i], map_res_);
            
            if (col>=0 && col<w && row>=0 && row<h && latest_costmap_.data[row*w+col] >= 80) {
                if (++hit_count >= 3) return true;
            }
        }
        return false;
    }

    float semantic_path_cost(const Eigen::ArrayXf& path_x, const Eigen::ArrayXf& path_y) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (!has_semantic_) return 0.0f;
        int w = latest_semantic_.info.width, h = latest_semantic_.info.height;
        float total = 0.0f; int n = path_x.size();
        for (int i = 0; i < n; ++i) {
            int col = sem_cx_ - round_idx(path_y[i], sem_res_);
            int row = sem_cy_ - round_idx(path_x[i], sem_res_);
            if (col<0 || col>=w || row<0 || row>=h) { 
                total += 50.0f; 
                continue; 
            } 
            
            int8_t v = latest_semantic_.data[row*w+col];
            
            if (v < 0) {
                total += 50.0f;  
            } else if (v >= 80) {
                total += 500.0f; 
            } else if (v >= 50) {
                total += 10.0f;  
            } else {
                total += 0.0f;   
            }
        }
        return n > 0 ? total / n : 0.0f;
    }

    void plan_local_path() {
        if (!has_odom_) return;

        std::vector<std::pair<float, float>> g_path;
        nav_msgs::msg::Odometry odom;
        float tgt_offset;
        
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            g_path = global_path_; odom = latest_odom_; tgt_offset = target_lane_offset_;
        }

        if (g_path.empty()) return;

        float ego_x = odom.pose.pose.position.x, ego_y = odom.pose.pose.position.y;
        float ego_yaw = get_ego_yaw(odom);
        float cos_yaw = std::cos(ego_yaw), sin_yaw = std::sin(ego_yaw);

        // DYNAMIC CORNERING DETECTION
        float lookahead_yaw_diff = 0.0f;
        float dist_accum = 0.0f;
        for (size_t i = last_closest_idx_ + 1; i < g_path.size(); ++i) {
            dist_accum += std::hypot(g_path[i].first - g_path[i-1].first, g_path[i].second - g_path[i-1].second);
            if (dist_accum > 10.0f) {
                float path_yaw = std::atan2(-(g_path[i].second - g_path[i-1].second), g_path[i].first - g_path[i-1].first);
                float diff = path_yaw - ego_yaw;
                while (diff > M_PI) diff -= 2.0f * M_PI;
                while (diff < -M_PI) diff += 2.0f * M_PI;
                lookahead_yaw_diff = std::abs(diff);
                break;
            }
        }

        bool is_turning_now = (lookahead_yaw_diff > 0.3f);
        bool just_finished_turn = (was_turning_ && !is_turning_now);

        size_t search_window = just_finished_turn ? 400 : 150;
        size_t search_end = std::min(last_closest_idx_ + search_window, g_path.size());
        if (last_closest_idx_ == 0) search_end = std::min(static_cast<size_t>(250), g_path.size());

        if (just_finished_turn) {
            last_chosen_offset_ = 0.0f; 
        }
        was_turning_ = is_turning_now;

        // --- FIND CLOSEST WAYPOINT ---
        size_t best_idx = last_closest_idx_;
        float min_dist = std::numeric_limits<float>::max();
        
        for (size_t i = last_closest_idx_; i < search_end; ++i) {
            float dist = std::hypot(g_path[i].first - ego_x, g_path[i].second - ego_y);
            
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = i;
            }
        }

        // Global Re-anchor Fallback
        if (min_dist > 8.0f) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Global path lost sync (nearest=%.1fm) — full re-anchor", min_dist);
            for (size_t i = 0; i < g_path.size(); ++i) {
                float dist = std::hypot(g_path[i].first - ego_x, g_path[i].second - ego_y);
                if (dist < min_dist) { min_dist = dist; best_idx = i; }
            }
            last_closest_idx_ = best_idx;
            last_chosen_offset_ = 0.0f;  
            was_turning_ = false;
        }

        last_closest_idx_ = best_idx;

        std::vector<float> local_x = {0.0f}, local_y = {0.0f};

        // --- DYNAMIC LOOKAHEAD & SPACING ---
        float dynamic_target_dist = target_dist_; 
        float pt_spacing = 1.5f;                 

        if (lookahead_yaw_diff > 0.3f) {
            dynamic_target_dist = std::max(3.5f, 12.0f - (lookahead_yaw_diff * 6.0f));
            pt_spacing = 0.5f; 
        }

        // --- PROJECTION TO LOCAL FRAME ---
        for (size_t i = best_idx; i < g_path.size(); ++i) {
            const auto& pt = g_path[i];
            float dx = pt.first - ego_x;
            
            float dy_rh = -(pt.second - ego_y); 
            
            float lx = dx * cos_yaw + dy_rh * sin_yaw;
            float ly = -dx * sin_yaw + dy_rh * cos_yaw;
            
            if (lx <= -1.0f) continue; 
            
            if (std::hypot(lx - local_x.back(), ly - local_y.back()) >= pt_spacing) {
                local_x.push_back(lx); local_y.push_back(ly);
            }
            
            if (local_x.back() > lookahead_dist_ + 10.0f) break;
        }

        if (local_x.size() < 2) { local_x.push_back(lookahead_dist_); local_y.push_back(0.0f); }

        // Lateral Coherence Check
        bool ref_is_sane = true;
        
        if (!is_turning_now) {
            for (size_t i = 1; i < std::min(local_x.size(), (size_t)6); ++i) {
                if (std::abs(local_y[i]) > 5.0f) { ref_is_sane = false; break; }
            }
            if (!ref_is_sane) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "Reference diverges (lateral > 5m) — straight fallback");
                local_x = {0.0f, 8.0f, 16.0f, lookahead_dist_};
                local_y = {0.0f, 0.0f, 0.0f, 0.0f};
                last_chosen_offset_ = 0.0f;
            }
        }

        std::vector<float> s_vec = {0.0f};
        for (size_t i = 1; i < local_x.size(); ++i) s_vec.push_back(s_vec.back() + std::hypot(local_x[i] - local_x[i-1], local_y[i] - local_y[i-1]));

        if (s_vec.back() < lookahead_dist_) {
            float extra_s = lookahead_dist_ - s_vec.back();
            float dx = local_x.back() - local_x[local_x.size() - 2];
            float dy = local_y.back() - local_y[local_y.size() - 2];
            float mag = std::hypot(dx, dy) + 1e-6f;
            local_x.push_back(local_x.back() + (dx / mag) * extra_s);
            local_y.push_back(local_y.back() + (dy / mag) * extra_s);
            s_vec.push_back(s_vec.back() + extra_s);
        }

        std::vector<float> uniq_s, uniq_x, uniq_y;
        uniq_s.push_back(s_vec[0]); uniq_x.push_back(local_x[0]); uniq_y.push_back(local_y[0]);
        for (size_t i = 1; i < s_vec.size(); ++i) {
            if (s_vec[i] - uniq_s.back() > 1e-3f) {
                uniq_s.push_back(s_vec[i]); uniq_x.push_back(local_x[i]); uniq_y.push_back(local_y[i]);
            }
        }

        if (uniq_s.size() == 2) {
            uniq_s.insert(uniq_s.begin() + 1, (uniq_s[0] + uniq_s[1]) / 2.0f);
            uniq_x.insert(uniq_x.begin() + 1, (uniq_x[0] + uniq_x[1]) / 2.0f);
            uniq_y.insert(uniq_y.begin() + 1, (uniq_y[0] + uniq_y[1]) / 2.0f);
        } else if (uniq_s.size() < 2) {
            return;
        }

        CubicSpline ref_x_spline(uniq_s, uniq_x), ref_y_spline(uniq_s, uniq_y);
        if (!ref_x_spline.is_valid() || !ref_y_spline.is_valid()) return;

        int num_eval = 50;
        Eigen::ArrayXf s_eval = Eigen::ArrayXf::LinSpaced(num_eval, 0.0f, lookahead_dist_);
        Eigen::ArrayXf ref_x(num_eval), ref_y(num_eval), nx(num_eval), ny(num_eval);

        for (int i = 0; i < num_eval; ++i) {
            float s = s_eval[i];
            ref_x[i] = ref_x_spline.calc(s); ref_y[i] = ref_y_spline.calc(s);
            float dx_ds = ref_x_spline.calc_deriv(s), dy_ds = ref_y_spline.calc_deriv(s);
            float mag = std::hypot(dx_ds, dy_ds) + 1e-6f;
            nx[i] = -dy_ds / mag; ny[i] = dx_ds / mag;
        }

        Eigen::ArrayXf best_x, best_y;
        float best_cost = std::numeric_limits<float>::infinity(), best_offset = tgt_offset;
        bool path_found = false;
        int blocked_count = 0;
        bool straight_blocked = false;

        Eigen::ArrayXf t = s_eval / lookahead_dist_;
        Eigen::ArrayXf poly = 10.0f * t.pow(3) - 15.0f * t.pow(4) + 6.0f * t.pow(5);

        for (float delta : lattice_deltas_) {
            float offset = std::clamp(tgt_offset + delta, -lane_offset_abs_limit_, lane_offset_abs_limit_);
            Eigen::ArrayXf d_eval = offset * poly;
            Eigen::ArrayXf test_x = ref_x + d_eval * nx;
            Eigen::ArrayXf test_y = ref_y + d_eval * ny;

            if (is_hard_blocked(test_x, test_y)) { 
                blocked_count++; 
                if (delta == 0.0f) straight_blocked = true; 
                continue; 
            }

            float target_cost = std::abs(offset - tgt_offset) * 2.0f;
            float continuity_cost = std::abs(offset - last_chosen_offset_) * 1.0f; 
            float drivability_cost = semantic_path_cost(test_x, test_y) * 0.15f; 
            
            float cost = target_cost + continuity_cost + drivability_cost;

            if (cost < best_cost) { 
                best_cost = cost; best_x = test_x; best_y = test_y; 
                best_offset = offset; path_found = true; 
            }
        }

        int target_idx = 0;
        
        if (!path_found && is_turning_now) {
            best_x = ref_x;
            best_y = ref_y;
            best_offset = tgt_offset;
            path_found = true;
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 500, 
                "Mid-turn block detected: Forcing center-line fallback to prevent swerve!");
        }

        std_msgs::msg::Bool hazard_msg;

        if (path_found) {
            last_chosen_offset_ = best_offset;
            float min_diff = std::numeric_limits<float>::max();
            for(int i = 0; i < num_eval; ++i){
                float diff = std::abs(s_eval[i] - dynamic_target_dist);
                if(diff < min_diff){ min_diff = diff; target_idx = i; }
            }
            
            auto current_time = this->get_clock()->now();

            // Convert local lattice point to global map frame
            float lx = best_x[target_idx];
            float ly = best_y[target_idx];
            float dx_global = lx * cos_yaw - ly * sin_yaw;
            float dy_rh    = lx * sin_yaw + ly * cos_yaw;
            float map_x = ego_x + dx_global;
            float map_y = ego_y - dy_rh;  // re-applies the RH flip

            geometry_msgs::msg::PoseStamped wp_msg;
            wp_msg.header.stamp = current_time;
            wp_msg.header.frame_id = "map";  // NOT base_link
            wp_msg.pose.position.x = map_x;
            wp_msg.pose.position.y = map_y;
            wp_msg.pose.orientation.w = 1.0f;
            wp_pub_->publish(wp_msg);

            nav_msgs::msg::Path path_msg;
            path_msg.header.stamp = current_time; path_msg.header.frame_id = "base_link";
            for (int i = 0; i < num_eval; ++i) {
                geometry_msgs::msg::PoseStamped p;
                p.pose.position.x = best_x[i]; p.pose.position.y = best_y[i];
                path_msg.poses.push_back(p);
            }
            path_pub_->publish(path_msg);
        }

        lat_log_ << std::fixed << std::setprecision(3) 
                 << this->get_clock()->now().seconds() << "," << ego_x << "," << ego_y << "," << ego_yaw << ","
                 << tgt_offset << "," << (path_found ? best_offset : std::nan("")) << ","
                 << best_cost << "," << blocked_count << "," << (straight_blocked ? "1" : "0") << ","
                 << (path_found ? best_x[target_idx] : 0.0f) << "," << (path_found ? best_y[target_idx] : 0.0f) << ","
                 << dynamic_target_dist << "," << min_dist << "," << (ref_is_sane ? "1" : "0") << "\n";
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LatticePlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}