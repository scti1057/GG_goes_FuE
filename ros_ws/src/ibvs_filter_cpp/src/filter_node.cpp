#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/twist.hpp"
#include "ibvs_filter_cpp/filters.hpp"
#include "ibvs_msgs/msg/keypoints.hpp"
#include "ibvs_msgs/msg/matches.hpp"
#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/callback_group.hpp"
#include "rclcpp/executors/multi_threaded_executor.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/u_int32.hpp"

namespace
{

class FilterNode : public rclcpp::Node
{
public:
  FilterNode()
  : Node("ibvs_filter_node"),
    has_k_(false),
    update_step_count_(0),
    update_success_count_(0),
    last_update_status_("NO UPDATE YET"),
    latest_camera_velocity_(Eigen::Matrix<double, 6, 1>::Zero()),
    has_velocity_stamp_(false),
    has_last_predict_time_(false)
  {
    declareParameters();
    loadParameters();

    cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    rclcpp::SubscriptionOptions sub_options;
    sub_options.callback_group = cb_group_;

    sub_cam_info_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera/camera/color/camera_info",
      10,
      std::bind(&FilterNode::camInfoCallback, this, std::placeholders::_1),
      sub_options);

    sub_ref_ = create_subscription<ibvs_msgs::msg::Keypoints>(
      "/ibvs/reference/keypoints",
      10,
      std::bind(&FilterNode::referenceCallback, this, std::placeholders::_1),
      sub_options);

    sub_camera_velocity_ = create_subscription<geometry_msgs::msg::Twist>(
      camera_velocity_topic_,
      10,
      std::bind(&FilterNode::cameraVelocityCallback, this, std::placeholders::_1),
      sub_options);

    sub_matches_ = create_subscription<ibvs_msgs::msg::Matches>(
      "/ibvs/matches",
      10,
      std::bind(&FilterNode::matchesCallback, this, std::placeholders::_1),
      sub_options);

    pub_filtered_points_ = create_publisher<ibvs_msgs::msg::Matches>("/ibvs/filtered_features", 10);
    pub_filter_status_ = create_publisher<std_msgs::msg::String>(filter_status_topic_, 10);
    pub_filter_uncertainty_ = create_publisher<std_msgs::msg::Float32>(filter_uncertainty_topic_, 10);
    pub_filter_update_status_ =
      create_publisher<std_msgs::msg::String>(filter_update_status_topic_, 10);
    pub_filter_update_count_ =
      create_publisher<std_msgs::msg::UInt32>(filter_update_count_topic_, 10);
    pub_filter_update_success_count_ =
      create_publisher<std_msgs::msg::UInt32>(filter_update_success_count_topic_, 10);
    pub_active_count_ = create_publisher<std_msgs::msg::UInt32>(active_count_topic_, 10);

    resetPredictTimer();

    parameters_callback_handle_ = add_on_set_parameters_callback(
      std::bind(&FilterNode::onParametersChanged, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "Filter Node gestartet. Modus: %s. Warte auf K-Matrix und Referenz...",
      filter_type_.c_str());
  }

private:
  void declareParameters()
  {
    declare_parameter<std::string>("filter_type", "ekf");
    declare_parameter<double>("q_noise", 2.0);
    declare_parameter<double>("r_noise", 1.1);
    declare_parameter<double>("z_depth", 0.25);
    declare_parameter<bool>("use_depth_from_matches", true);
    declare_parameter<double>("depth_min_valid_m", 0.05);
    declare_parameter<double>("depth_max_valid_m", 3.0);
    declare_parameter<double>("depth_ema_alpha", 0.35);
    declare_parameter<double>("gate_threshold", 20.0);
    declare_parameter<double>("predict_rate", 120.0);

    declare_parameter<int64_t>("max_active_keypoints", 10);
    declare_parameter<int64_t>("min_init_keypoints", 8);
    declare_parameter<int64_t>("min_update_keypoints", 4);
    declare_parameter<bool>("force_relocalization", false);

    declare_parameter<std::string>("base_frame", "base_link");
    declare_parameter<std::string>("camera_frame", "camera_color_frame");
    declare_parameter<std::string>(
      "camera_velocity_topic",
      "/cartesian_twist_passthrough_controller/cmd_vel");
    declare_parameter<double>("camera_velocity_deadband_linear", 0.0);
    declare_parameter<double>("camera_velocity_deadband_angular", 0.0);
    declare_parameter<double>("camera_velocity_stale_timeout", 0.2);

    declare_parameter<std::string>("filter_status_topic", "/ibvs/filter/status");
    declare_parameter<std::string>("filter_uncertainty_topic", "/ibvs/filter/uncertainty");
    declare_parameter<std::string>("filter_update_status_topic", "/ibvs/filter/update_status");
    declare_parameter<std::string>("filter_update_count_topic", "/ibvs/filter/update_count");
    declare_parameter<std::string>(
      "filter_update_success_count_topic",
      "/ibvs/filter/update_success_count");
    declare_parameter<std::string>("active_count_topic", "/ibvs/filter/active_count");
  }

  void loadParameters()
  {
    filter_type_ = get_parameter("filter_type").as_string();
    q_noise_ = get_parameter("q_noise").as_double();
    r_noise_ = get_parameter("r_noise").as_double();
    gate_threshold_ = get_parameter("gate_threshold").as_double();
    z_depth_ = get_parameter("z_depth").as_double();
    use_depth_from_matches_ = get_parameter("use_depth_from_matches").as_bool();
    depth_min_valid_m_ = get_parameter("depth_min_valid_m").as_double();
    depth_max_valid_m_ = get_parameter("depth_max_valid_m").as_double();
    depth_ema_alpha_ = get_parameter("depth_ema_alpha").as_double();
    predict_rate_ = get_parameter("predict_rate").as_double();
    if (!has_runtime_depth_) {
      runtime_z_depth_ = z_depth_;
    }

    max_active_keypoints_ = static_cast<int>(get_parameter("max_active_keypoints").as_int());
    min_init_keypoints_ = static_cast<int>(get_parameter("min_init_keypoints").as_int());
    min_update_keypoints_ = static_cast<int>(get_parameter("min_update_keypoints").as_int());
    force_relocalization_param_ = get_parameter("force_relocalization").as_bool();

    base_frame_ = get_parameter("base_frame").as_string();
    camera_frame_ = get_parameter("camera_frame").as_string();
    camera_velocity_topic_ = get_parameter("camera_velocity_topic").as_string();
    camera_velocity_deadband_linear_ = get_parameter("camera_velocity_deadband_linear").as_double();
    camera_velocity_deadband_angular_ = get_parameter("camera_velocity_deadband_angular").as_double();
    camera_velocity_stale_timeout_ = get_parameter("camera_velocity_stale_timeout").as_double();

    filter_status_topic_ = get_parameter("filter_status_topic").as_string();
    filter_uncertainty_topic_ = get_parameter("filter_uncertainty_topic").as_string();
    filter_update_status_topic_ = get_parameter("filter_update_status_topic").as_string();
    filter_update_count_topic_ = get_parameter("filter_update_count_topic").as_string();
    filter_update_success_count_topic_ =
      get_parameter("filter_update_success_count_topic").as_string();
    active_count_topic_ = get_parameter("active_count_topic").as_string();
  }

  void camInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    if (has_k_) {
      return;
    }

    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        K_(r, c) = msg->k[3 * r + c];
      }
    }

    has_k_ = true;
    initFilter();
    RCLCPP_INFO(get_logger(), "CameraInfo empfangen!");
  }

  void initFilter()
  {
    if (filter_type_ == "ekf") {
      filter_ = std::make_unique<ibvs_filter_cpp::ExtendedKalmanFilter>(K_);
    } else if (filter_type_ == "ukf") {
      filter_ = std::make_unique<ibvs_filter_cpp::UnscentedKalmanFilter>(K_);
    } else if (filter_type_ == "eskf") {
      filter_ = std::make_unique<ibvs_filter_cpp::ErrorStateKalmanFilter>(K_);
    } else if (filter_type_ == "skf") {
      filter_ = std::make_unique<ibvs_filter_cpp::StandardKalmanFilter>(K_);
    } else {
      RCLCPP_ERROR(get_logger(), "Unbekannter Filtertyp: %s", filter_type_.c_str());
      return;
    }

    filter_->setQRGate(q_noise_, r_noise_, gate_threshold_);
    filter_->configureKeypointTracking(
      max_active_keypoints_,
      min_init_keypoints_,
      min_update_keypoints_);

    RCLCPP_INFO(
      get_logger(),
      "Filter initialized. active_max=%d, min_init=%d, min_update=%d",
      max_active_keypoints_,
      min_init_keypoints_,
      min_update_keypoints_);
  }

  rcl_interfaces::msg::SetParametersResult onParametersChanged(
    const std::vector<rclcpp::Parameter> & params)
  {
    auto result = rcl_interfaces::msg::SetParametersResult();
    result.successful = true;

    double next_q = q_noise_;
    double next_r = r_noise_;
    double next_gate = gate_threshold_;
    double next_z = z_depth_;
    bool next_use_depth_from_matches = use_depth_from_matches_;
    double next_depth_min_valid_m = depth_min_valid_m_;
    double next_depth_max_valid_m = depth_max_valid_m_;
    double next_depth_ema_alpha = depth_ema_alpha_;
    double next_predict_rate = predict_rate_;

    int next_active_max = max_active_keypoints_;
    int next_min_init = min_init_keypoints_;
    int next_min_update = min_update_keypoints_;
    bool next_force_relocalization = force_relocalization_param_;

    double next_deadband_lin = camera_velocity_deadband_linear_;
    double next_deadband_ang = camera_velocity_deadband_angular_;
    double next_stale_timeout = camera_velocity_stale_timeout_;

    bool relocalization_requested = false;

    for (const auto & p : params) {
      if (p.get_name() == "filter_type") {
        result.successful = false;
        result.reason = "filter_type cannot be changed at runtime. Restart node.";
        return result;
      }

      if (p.get_name() == "q_noise") {
        if (p.as_double() <= 0.0) {
          result.successful = false;
          result.reason = "q_noise must be > 0";
          return result;
        }
        next_q = p.as_double();
      } else if (p.get_name() == "r_noise") {
        if (p.as_double() <= 0.0) {
          result.successful = false;
          result.reason = "r_noise must be > 0";
          return result;
        }
        next_r = p.as_double();
      } else if (p.get_name() == "gate_threshold") {
        if (p.as_double() <= 0.0) {
          result.successful = false;
          result.reason = "gate_threshold must be > 0";
          return result;
        }
        next_gate = p.as_double();
      } else if (p.get_name() == "z_depth") {
        if (p.as_double() <= 0.0) {
          result.successful = false;
          result.reason = "z_depth must be > 0";
          return result;
        }
        next_z = p.as_double();
      } else if (p.get_name() == "use_depth_from_matches") {
        next_use_depth_from_matches = p.as_bool();
      } else if (p.get_name() == "depth_min_valid_m") {
        if (p.as_double() <= 0.0) {
          result.successful = false;
          result.reason = "depth_min_valid_m must be > 0";
          return result;
        }
        next_depth_min_valid_m = p.as_double();
      } else if (p.get_name() == "depth_max_valid_m") {
        if (p.as_double() <= 0.0) {
          result.successful = false;
          result.reason = "depth_max_valid_m must be > 0";
          return result;
        }
        next_depth_max_valid_m = p.as_double();
      } else if (p.get_name() == "depth_ema_alpha") {
        if (p.as_double() < 0.0 || p.as_double() > 1.0) {
          result.successful = false;
          result.reason = "depth_ema_alpha must be in [0, 1]";
          return result;
        }
        next_depth_ema_alpha = p.as_double();
      } else if (p.get_name() == "predict_rate") {
        if (p.as_double() <= 0.0) {
          result.successful = false;
          result.reason = "predict_rate must be > 0";
          return result;
        }
        next_predict_rate = p.as_double();
      } else if (p.get_name() == "max_active_keypoints") {
        if (p.as_int() < 4) {
          result.successful = false;
          result.reason = "max_active_keypoints must be >= 4";
          return result;
        }
        next_active_max = static_cast<int>(p.as_int());
      } else if (p.get_name() == "min_init_keypoints") {
        if (p.as_int() < 4) {
          result.successful = false;
          result.reason = "min_init_keypoints must be >= 4";
          return result;
        }
        next_min_init = static_cast<int>(p.as_int());
      } else if (p.get_name() == "min_update_keypoints") {
        if (p.as_int() < 0) {
          result.successful = false;
          result.reason = "min_update_keypoints must be >= 0";
          return result;
        }
        next_min_update = static_cast<int>(p.as_int());
      } else if (p.get_name() == "force_relocalization") {
        next_force_relocalization = p.as_bool();
        if (next_force_relocalization) {
          relocalization_requested = true;
        }
      } else if (p.get_name() == "camera_velocity_deadband_linear") {
        if (p.as_double() < 0.0) {
          result.successful = false;
          result.reason = "camera_velocity_deadband_linear must be >= 0";
          return result;
        }
        next_deadband_lin = p.as_double();
      } else if (p.get_name() == "camera_velocity_deadband_angular") {
        if (p.as_double() < 0.0) {
          result.successful = false;
          result.reason = "camera_velocity_deadband_angular must be >= 0";
          return result;
        }
        next_deadband_ang = p.as_double();
      } else if (p.get_name() == "camera_velocity_stale_timeout") {
        if (p.as_double() <= 0.0) {
          result.successful = false;
          result.reason = "camera_velocity_stale_timeout must be > 0";
          return result;
        }
        next_stale_timeout = p.as_double();
      }
    }

    if (next_depth_max_valid_m <= next_depth_min_valid_m) {
      result.successful = false;
      result.reason = "depth_max_valid_m must be > depth_min_valid_m";
      return result;
    }

    q_noise_ = next_q;
    r_noise_ = next_r;
    gate_threshold_ = next_gate;
    z_depth_ = next_z;
    use_depth_from_matches_ = next_use_depth_from_matches;
    depth_min_valid_m_ = next_depth_min_valid_m;
    depth_max_valid_m_ = next_depth_max_valid_m;
    depth_ema_alpha_ = next_depth_ema_alpha;

    const bool rate_changed = std::abs(next_predict_rate - predict_rate_) > 1e-12;
    predict_rate_ = next_predict_rate;

    max_active_keypoints_ = next_active_max;
    min_init_keypoints_ = next_min_init;
    min_update_keypoints_ = next_min_update;
    force_relocalization_param_ = next_force_relocalization;

    camera_velocity_deadband_linear_ = next_deadband_lin;
    camera_velocity_deadband_angular_ = next_deadband_ang;
    camera_velocity_stale_timeout_ = next_stale_timeout;
    if (!has_runtime_depth_) {
      runtime_z_depth_ = z_depth_;
    }

    if (filter_ != nullptr) {
      std::lock_guard<std::mutex> lock(lock_);
      filter_->setQRGate(q_noise_, r_noise_, gate_threshold_);
      filter_->configureKeypointTracking(
        max_active_keypoints_,
        min_init_keypoints_,
        min_update_keypoints_);

      if (relocalization_requested) {
        filter_->forceRelocalization();
        last_update_status_ = "RELOCALIZATION REQUESTED";
        RCLCPP_WARN(
          get_logger(),
          "Manual relocalization requested via parameter force_relocalization=true");
      }
    } else if (relocalization_requested) {
      RCLCPP_WARN(get_logger(), "force_relocalization requested, but filter is not initialized yet");
    }

    if (rate_changed) {
      resetPredictTimer();
    }

    RCLCPP_INFO(
      get_logger(),
      "Tuning updated: q=%.4f, r=%.4f, gate=%.4f, z=%.4f, active_max=%d, min_init=%d, min_update=%d",
      q_noise_,
      r_noise_,
      gate_threshold_,
      z_depth_,
      max_active_keypoints_,
      min_init_keypoints_,
      min_update_keypoints_);

    return result;
  }

  void referenceCallback(const ibvs_msgs::msg::Keypoints::SharedPtr msg)
  {
    if (has_reference_) {
      return;
    }

    reference_keypoints_raw_.assign(msg->xy.begin(), msg->xy.end());
    has_reference_ = true;

    RCLCPP_INFO(get_logger(), "Neue Referenz empfangen! (%zu Keypoints)", msg->xy.size() / 2);

    if (filter_ != nullptr) {
      std::lock_guard<std::mutex> lock(lock_);
      filter_->forceRelocalization();
    }
  }

  void cameraVelocityCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    Eigen::Matrix<double, 6, 1> twist;
    twist <<
      msg->linear.x,
      msg->linear.y,
      msg->linear.z,
      msg->angular.x,
      msg->angular.y,
      msg->angular.z;

    for (int i = 0; i < 3; ++i) {
      if (std::abs(twist(i)) < camera_velocity_deadband_linear_) {
        twist(i) = 0.0;
      }
      if (std::abs(twist(i + 3)) < camera_velocity_deadband_angular_) {
        twist(i + 3) = 0.0;
      }
    }

    std::lock_guard<std::mutex> pose_lock(pose_lock_);
    latest_camera_velocity_ = twist;
    latest_velocity_stamp_ = now();
    has_velocity_stamp_ = true;
  }

  Eigen::Matrix<double, 6, 1> getCameraVelocityFromTopic()
  {
    Eigen::Matrix<double, 6, 1> twist;
    rclcpp::Time stamp;
    bool has_stamp = false;

    {
      std::lock_guard<std::mutex> pose_lock(pose_lock_);
      twist = latest_camera_velocity_;
      stamp = latest_velocity_stamp_;
      has_stamp = has_velocity_stamp_;
    }

    if (!has_stamp) {
      return Eigen::Matrix<double, 6, 1>::Zero();
    }

    const double age = (now() - stamp).seconds();
    if (age > camera_velocity_stale_timeout_) {
      return Eigen::Matrix<double, 6, 1>::Zero();
    }
    return twist;
  }

  double getPredictDt()
  {
    const rclcpp::Time now_t = now();

    if (!has_last_predict_time_) {
      last_predict_time_ = now_t;
      has_last_predict_time_ = true;
      return 1.0 / std::max(predict_rate_, 1e-3);
    }

    const double dt = (now_t - last_predict_time_).seconds();
    last_predict_time_ = now_t;

    if (dt <= 0.0 || dt > 1.0) {
      return 1.0 / std::max(predict_rate_, 1e-3);
    }

    return dt;
  }

  void publishFilterMeta(double p_trace)
  {
    if (filter_ == nullptr) {
      return;
    }

    std_msgs::msg::String status_msg;
    status_msg.data = toUpper(filter_type_) + " | " + filter_->getStatus();
    pub_filter_status_->publish(status_msg);

    std_msgs::msg::Float32 unc_msg;
    unc_msg.data = static_cast<float>(std::max(0.0, p_trace));
    pub_filter_uncertainty_->publish(unc_msg);

    std_msgs::msg::String upd_status_msg;
    upd_status_msg.data = last_update_status_;
    pub_filter_update_status_->publish(upd_status_msg);

    std_msgs::msg::UInt32 upd_count_msg;
    upd_count_msg.data = static_cast<uint32_t>(std::max(0u, update_step_count_));
    pub_filter_update_count_->publish(upd_count_msg);

    std_msgs::msg::UInt32 upd_success_msg;
    upd_success_msg.data = static_cast<uint32_t>(std::max(0u, update_success_count_));
    pub_filter_update_success_count_->publish(upd_success_msg);

    std_msgs::msg::UInt32 active_count_msg;
    active_count_msg.data = static_cast<uint32_t>(std::max(0, filter_->getActiveCount()));
    pub_active_count_->publish(active_count_msg);
  }

  static std::string toUpper(std::string in)
  {
    std::transform(in.begin(), in.end(), in.begin(), [](unsigned char ch) {return std::toupper(ch);});
    return in;
  }

  static std::vector<float> extractActivePositionUncertainty(
    const Eigen::MatrixXd & p_mat,
    size_t n)
  {
    std::vector<float> sigma_px(n, 0.0f);

    for (size_t slot = 0; slot < n; ++slot) {
      const int i0 = static_cast<int>(2 * slot);
      const int i1 = i0 + 2;
      if (i1 <= p_mat.rows() && i1 <= p_mat.cols()) {
        const Eigen::Matrix2d p_block = p_mat.block<2, 2>(i0, i0);
        double tr = p_block.trace();
        if (!std::isfinite(tr)) {
          tr = 0.0;
        }
        sigma_px[slot] = static_cast<float>(std::sqrt(std::max(0.0, tr)));
      }
    }

    return sigma_px;
  }

  static double medianOf(std::vector<double> v)
  {
    if (v.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const size_t mid = v.size() / 2U;
    std::nth_element(v.begin(), v.begin() + static_cast<long>(mid), v.end());
    double med = v[mid];
    if ((v.size() % 2U) == 0U && mid > 0U) {
      std::nth_element(v.begin(), v.begin() + static_cast<long>(mid - 1U), v.end());
      med = 0.5 * (med + v[mid - 1U]);
    }
    return med;
  }

  void timerCallback()
  {
    if (filter_ == nullptr || !has_reference_) {
      return;
    }

    const double dt = getPredictDt();
    const Eigen::Matrix<double, 6, 1> v_ee = getCameraVelocityFromTopic();

    std::vector<int64_t> active_ref_ids;
    Eigen::MatrixXd filtered_current_pts;
    std::vector<float> active_sigma_px;
    std::vector<float> active_depth_m;
    double p_trace = 0.0;
    double predict_z = z_depth_;

    {
      std::lock_guard<std::mutex> lock(lock_);
      if (
        use_depth_from_matches_ &&
        has_runtime_depth_ &&
        std::isfinite(runtime_z_depth_) &&
        runtime_z_depth_ > 0.0)
      {
        predict_z = runtime_z_depth_;
      }

      filter_->predict(v_ee, predict_z, dt);
      active_ref_ids = filter_->getActiveRefIds();
      filtered_current_pts = filter_->getActiveFilteredPoints();

      const Eigen::MatrixXd cov = filter_->getCovariance();
      active_sigma_px = extractActivePositionUncertainty(cov, active_ref_ids.size());
      if (cov.rows() > 0 && cov.cols() > 0) {
        p_trace = cov.trace();
      }

      active_depth_m.reserve(active_ref_ids.size());
      for (const int64_t rid : active_ref_ids) {
        float z = static_cast<float>(predict_z);
        auto it = latest_depth_by_ref_.find(rid);
        if (it != latest_depth_by_ref_.end()) {
          const float cand = it->second;
          if (
            std::isfinite(cand) &&
            cand > static_cast<float>(depth_min_valid_m_) &&
            cand < static_cast<float>(depth_max_valid_m_))
          {
            z = cand;
          }
        }
        active_depth_m.push_back(z);
      }
    }

    ibvs_msgs::msg::Matches out_msg;
    out_msg.header.stamp = now();
    out_msg.header.frame_id = camera_frame_;

    const size_t num_pts = std::min(
      static_cast<size_t>(filtered_current_pts.cols()),
      active_ref_ids.size());

    if (num_pts > 0) {
      out_msg.ref_id.reserve(num_pts);
      out_msg.xy.reserve(2 * num_pts);
      out_msg.depth_m.reserve(num_pts);
      out_msg.sim.reserve(num_pts);

      for (size_t i = 0; i < num_pts; ++i) {
        out_msg.ref_id.push_back(static_cast<uint32_t>(std::max<int64_t>(0, active_ref_ids[i])));
        out_msg.xy.push_back(static_cast<float>(filtered_current_pts(0, static_cast<int>(i))));
        out_msg.xy.push_back(static_cast<float>(filtered_current_pts(1, static_cast<int>(i))));
        out_msg.depth_m.push_back(active_depth_m[i]);
        out_msg.sim.push_back(active_sigma_px[i]);
      }
    }

    pub_filtered_points_->publish(out_msg);
    publishFilterMeta(p_trace);
  }

  void matchesCallback(const ibvs_msgs::msg::Matches::SharedPtr matches_msg)
  {
    if (filter_ == nullptr || !has_reference_) {
      return;
    }

    const int num_matches = static_cast<int>(matches_msg->ref_id.size());
    if (num_matches <= 0) {
      return;
    }

    Eigen::MatrixXd current_pixels = Eigen::MatrixXd::Zero(2, num_matches);
    Eigen::MatrixXd desired_pixels = Eigen::MatrixXd::Zero(2, num_matches);
    std::vector<int64_t> ref_ids(static_cast<size_t>(num_matches), 0);
    Eigen::VectorXd match_scores = Eigen::VectorXd::Zero(num_matches);

    int valid_count = 0;
    const int n_xy_pairs = static_cast<int>(matches_msg->xy.size() / 2);
    const int n_depth = static_cast<int>(matches_msg->depth_m.size());
    const int n_scores = static_cast<int>(matches_msg->sim.size());
    const int ref_raw_len = static_cast<int>(reference_keypoints_raw_.size());
    std::vector<double> valid_depth_samples;
    std::vector<std::pair<int64_t, float>> valid_depth_by_ref;

    for (int i = 0; i < num_matches; ++i) {
      const int64_t ref_idx = static_cast<int64_t>(matches_msg->ref_id[static_cast<size_t>(i)]);
      if (i >= n_xy_pairs) {
        break;
      }
      if (ref_idx < 0 || (2 * ref_idx + 1) >= ref_raw_len) {
        continue;
      }

      current_pixels(0, valid_count) = matches_msg->xy[static_cast<size_t>(2 * i)];
      current_pixels(1, valid_count) = matches_msg->xy[static_cast<size_t>(2 * i + 1)];
      desired_pixels(0, valid_count) = reference_keypoints_raw_[static_cast<size_t>(2 * ref_idx)];
      desired_pixels(1, valid_count) = reference_keypoints_raw_[static_cast<size_t>(2 * ref_idx + 1)];
      ref_ids[static_cast<size_t>(valid_count)] = ref_idx;

      if (i < n_scores) {
        match_scores(valid_count) = matches_msg->sim[static_cast<size_t>(i)];
      }

      if (i < n_depth) {
        const double z = static_cast<double>(matches_msg->depth_m[static_cast<size_t>(i)]);
        if (std::isfinite(z) && z > depth_min_valid_m_ && z < depth_max_valid_m_) {
          valid_depth_samples.push_back(z);
          valid_depth_by_ref.emplace_back(ref_idx, static_cast<float>(z));
        }
      }
      ++valid_count;
    }

    if (valid_count <= 0) {
      return;
    }

    current_pixels.conservativeResize(2, valid_count);
    desired_pixels.conservativeResize(2, valid_count);
    ref_ids.resize(static_cast<size_t>(valid_count));
    match_scores.conservativeResize(valid_count);

    std::string update_status;
    {
      std::lock_guard<std::mutex> lock(lock_);
      for (const auto & it : valid_depth_by_ref) {
        latest_depth_by_ref_[it.first] = it.second;
      }

      if (use_depth_from_matches_ && !valid_depth_samples.empty()) {
        const double z_med = medianOf(valid_depth_samples);
        if (std::isfinite(z_med) && z_med > 0.0) {
          if (!has_runtime_depth_) {
            runtime_z_depth_ = z_med;
            has_runtime_depth_ = true;
          } else {
            runtime_z_depth_ =
              depth_ema_alpha_ * z_med + (1.0 - depth_ema_alpha_) * runtime_z_depth_;
          }
        }
      }
      filter_->update(current_pixels, desired_pixels, ref_ids, match_scores);
      update_status = filter_->getStatus();
    }

    ++update_step_count_;
    last_update_status_ = update_status;
    if (
      update_status == "UPDATE" ||
      update_status == "INIT" ||
      update_status == "RELOCALIZED")
    {
      ++update_success_count_;
    }
  }

  void resetPredictTimer()
  {
    const double period_s = 1.0 / std::max(predict_rate_, 1e-3);
    const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(period_s));

    if (timer_ != nullptr) {
      timer_->cancel();
    }

    timer_ = create_wall_timer(
      period_ns,
      std::bind(&FilterNode::timerCallback, this),
      cb_group_);
  }

  std::string filter_type_;
  double q_noise_;
  double r_noise_;
  double gate_threshold_;
  double z_depth_;
  bool use_depth_from_matches_;
  double depth_min_valid_m_;
  double depth_max_valid_m_;
  double depth_ema_alpha_;
  double runtime_z_depth_ = 0.25;
  bool has_runtime_depth_ = false;
  double predict_rate_;

  int max_active_keypoints_;
  int min_init_keypoints_;
  int min_update_keypoints_;
  bool force_relocalization_param_;

  std::string base_frame_;
  std::string camera_frame_;
  std::string camera_velocity_topic_;
  double camera_velocity_deadband_linear_;
  double camera_velocity_deadband_angular_;
  double camera_velocity_stale_timeout_;

  std::string filter_status_topic_;
  std::string filter_uncertainty_topic_;
  std::string filter_update_status_topic_;
  std::string filter_update_count_topic_;
  std::string filter_update_success_count_topic_;
  std::string active_count_topic_;

  Eigen::Matrix3d K_ = Eigen::Matrix3d::Identity();
  bool has_k_;

  std::unique_ptr<ibvs_filter_cpp::BaseFilter> filter_;

  bool has_reference_ = false;
  std::vector<double> reference_keypoints_raw_;
  std::unordered_map<int64_t, float> latest_depth_by_ref_;

  uint32_t update_step_count_;
  uint32_t update_success_count_;
  std::string last_update_status_;

  std::mutex lock_;
  std::mutex pose_lock_;

  Eigen::Matrix<double, 6, 1> latest_camera_velocity_;
  rclcpp::Time latest_velocity_stamp_;
  bool has_velocity_stamp_;

  rclcpp::Time last_predict_time_;
  bool has_last_predict_time_;

  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameters_callback_handle_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_cam_info_;
  rclcpp::Subscription<ibvs_msgs::msg::Keypoints>::SharedPtr sub_ref_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_camera_velocity_;
  rclcpp::Subscription<ibvs_msgs::msg::Matches>::SharedPtr sub_matches_;

  rclcpp::Publisher<ibvs_msgs::msg::Matches>::SharedPtr pub_filtered_points_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_filter_status_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_filter_uncertainty_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_filter_update_status_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr pub_filter_update_count_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr pub_filter_update_success_count_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr pub_active_count_;

  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FilterNode>();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);

  try {
    executor.spin();
  } catch (const std::exception &) {
  }

  executor.remove_node(node);
  node.reset();
  rclcpp::shutdown();
  return 0;
}
