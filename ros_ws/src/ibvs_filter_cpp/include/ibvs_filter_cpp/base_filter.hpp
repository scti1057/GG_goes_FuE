#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ibvs_filter_cpp
{

struct MeasurementData
{
  Eigen::VectorXd z_k;
  std::vector<int> obs_slots;
  std::string prep_status;
};

class BaseFilter
{
public:
  explicit BaseFilter(const Eigen::Matrix3d & K);
  virtual ~BaseFilter() = default;

  virtual void reset();
  virtual void forceRelocalization();

  void configureKeypointTracking(
    int max_active_keypoints,
    int min_init_keypoints,
    int min_update_keypoints);

  int getActiveCount() const;
  std::vector<int64_t> getActiveRefIds() const;
  Eigen::MatrixXd getActiveFilteredPoints() const;

  bool updateGeometryFromState(const Eigen::VectorXd & state_2n, bool log_on_fail = false);

  const std::string & getStatus() const { return status_; }

  virtual void setQRGate(double q_val, double r_val, double gate_thresh_val) = 0;
  virtual void predict(const Eigen::Matrix<double, 6, 1> & v_ee, double z_est, double dt) = 0;
  virtual void update(
    const Eigen::MatrixXd & current_pixels,
    const Eigen::MatrixXd & desired_pixels,
    const std::vector<int64_t> & ref_ids,
    const Eigen::VectorXd & match_scores) = 0;

  virtual Eigen::MatrixXd getCovariance() const = 0;

protected:
  Eigen::Matrix<double, 6, 1> transformTwistEeToCam(const Eigen::Matrix<double, 6, 1> & v_ee) const;

  std::optional<MeasurementData> prepareMeasurement(
    const Eigen::MatrixXd & current_pixels,
    const Eigen::MatrixXd & desired_pixels,
    const std::vector<int64_t> & ref_ids,
    const Eigen::VectorXd & match_scores);

  virtual Eigen::VectorXd getActiveStateVector() const = 0;

  Eigen::Matrix3d K_;
  bool initialized_;
  std::string status_;
  Eigen::Matrix3d H_filtered_;

  std::vector<int64_t> active_ref_ids_;
  Eigen::MatrixXd active_desired_;

  int max_active_keypoints_;
  int min_init_keypoints_;
  int min_update_keypoints_;

private:
  std::vector<int> selectSpreadIndices(
    const Eigen::MatrixXd & desired_pixels,
    const Eigen::VectorXd & match_scores,
    int max_count) const;

  std::optional<Eigen::MatrixXd> initializeActiveSet(
    const Eigen::MatrixXd & current_pixels,
    const Eigen::MatrixXd & desired_pixels,
    const std::vector<int64_t> & ref_ids,
    const Eigen::VectorXd & match_scores);

  static Eigen::VectorXd stackMeasurement(const Eigen::MatrixXd & current_obs);
  bool checkGeometryAndUpdateH(const Eigen::VectorXd & state_2n, bool log_on_fail) ;
};

}  // namespace ibvs_filter_cpp
