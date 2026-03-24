#pragma once

#include <Eigen/Dense>

#include <string>
#include <vector>

#include "ibvs_filter_cpp/base_filter.hpp"
#include "ibvs_filter_cpp/ibvs_math.hpp"

namespace ibvs_filter_cpp
{

class ExtendedKalmanFilter : public BaseFilter
{
public:
  explicit ExtendedKalmanFilter(const Eigen::Matrix3d & K);

  void setQRGate(double q_val, double r_val, double gate_thresh_val) override;
  void forceRelocalization() override;
  void predict(
    const Eigen::Matrix<double, 6, 1> & v_ee,
    const Eigen::VectorXd & z_per_feature,
    double z_fallback,
    double dt) override;
  void update(
    const Eigen::MatrixXd & current_pixels,
    const Eigen::MatrixXd & desired_pixels,
    const std::vector<int64_t> & ref_ids,
    const Eigen::VectorXd & match_scores) override;
  Eigen::MatrixXd getCovariance() const override { return P_; }

protected:
  Eigen::VectorXd getActiveStateVector() const override;

private:
  void setStateDim(int active_count);
  Eigen::VectorXd computePixelVelocities(
    const Eigen::VectorXd & state_2n,
    const Eigen::Matrix<double, 6, 1> & v_cam,
    const Eigen::VectorXd & z_per_feature,
    double z_fallback) const;
  Eigen::MatrixXd buildObservationMatrix(const std::vector<int> & obs_slots) const;

  IBVSMath ibvs_math_;
  Eigen::VectorXd x_;
  Eigen::MatrixXd P_;
  Eigen::MatrixXd Q_;
  Eigen::MatrixXd R_;
  double q_noise_;
  double r_noise_;
  double gate_thresh_;
};

class UnscentedKalmanFilter : public BaseFilter
{
public:
  explicit UnscentedKalmanFilter(const Eigen::Matrix3d & K);

  void setQRGate(double q_val, double r_val, double gate_thresh_val) override;
  void forceRelocalization() override;
  void predict(
    const Eigen::Matrix<double, 6, 1> & v_ee,
    const Eigen::VectorXd & z_per_feature,
    double z_fallback,
    double dt) override;
  void update(
    const Eigen::MatrixXd & current_pixels,
    const Eigen::MatrixXd & desired_pixels,
    const std::vector<int64_t> & ref_ids,
    const Eigen::VectorXd & match_scores) override;
  Eigen::MatrixXd getCovariance() const override { return P_; }

protected:
  Eigen::VectorXd getActiveStateVector() const override;

private:
  void setStateDim(int active_count);
  void recomputeUtWeights();
  Eigen::VectorXd computePixelVelocities(
    const Eigen::VectorXd & state_2n,
    const Eigen::Matrix<double, 6, 1> & v_cam,
    const Eigen::VectorXd & z_per_feature,
    double z_fallback) const;
  Eigen::MatrixXd generateSigmaPoints(const Eigen::VectorXd & x, Eigen::MatrixXd P);
  Eigen::MatrixXd buildObservationMatrix(const std::vector<int> & obs_slots) const;

  IBVSMath ibvs_math_;
  int L_;
  Eigen::VectorXd x_;
  Eigen::MatrixXd P_;
  Eigen::MatrixXd Q_;
  Eigen::MatrixXd R_;
  double q_noise_;
  double r_noise_;
  double gate_thresh_;

  double alpha_;
  double beta_;
  double kappa_;
  double lambda_;
  Eigen::VectorXd Wm_;
  Eigen::VectorXd Wc_;
};

class ErrorStateKalmanFilter : public BaseFilter
{
public:
  explicit ErrorStateKalmanFilter(const Eigen::Matrix3d & K);

  void setQRGate(double q_val, double r_val, double gate_thresh_val) override;
  void forceRelocalization() override;
  void predict(
    const Eigen::Matrix<double, 6, 1> & v_ee,
    const Eigen::VectorXd & z_per_feature,
    double z_fallback,
    double dt) override;
  void update(
    const Eigen::MatrixXd & current_pixels,
    const Eigen::MatrixXd & desired_pixels,
    const std::vector<int64_t> & ref_ids,
    const Eigen::VectorXd & match_scores) override;
  Eigen::MatrixXd getCovariance() const override { return P_; }

protected:
  Eigen::VectorXd getActiveStateVector() const override;

private:
  void setStateDim(int active_count);
  Eigen::VectorXd computePixelVelocities(
    const Eigen::VectorXd & state_2n,
    const Eigen::Matrix<double, 6, 1> & v_cam,
    const Eigen::VectorXd & z_per_feature,
    double z_fallback) const;
  Eigen::MatrixXd buildObservationMatrix(const std::vector<int> & obs_slots) const;

  IBVSMath ibvs_math_;
  int L_;
  Eigen::VectorXd x_nom_;
  Eigen::VectorXd dx_;
  Eigen::MatrixXd P_;
  Eigen::MatrixXd Q_;
  Eigen::MatrixXd R_;
  double q_noise_;
  double r_noise_;
  double gate_thresh_;
};

class StandardKalmanFilter : public BaseFilter
{
public:
  explicit StandardKalmanFilter(const Eigen::Matrix3d & K);

  void setQRGate(double q_val, double r_val, double gate_thresh_val) override;
  void forceRelocalization() override;
  void predict(
    const Eigen::Matrix<double, 6, 1> & v_ee,
    const Eigen::VectorXd & z_per_feature,
    double z_fallback,
    double dt) override;
  void update(
    const Eigen::MatrixXd & current_pixels,
    const Eigen::MatrixXd & desired_pixels,
    const std::vector<int64_t> & ref_ids,
    const Eigen::VectorXd & match_scores) override;
  Eigen::MatrixXd getCovariance() const override { return P_; }

protected:
  Eigen::VectorXd getActiveStateVector() const override;

private:
  void setStateDim(int active_count);
  Eigen::MatrixXd buildObservationMatrix(const std::vector<int> & obs_slots) const;

  int keypoint_count_;
  int pos_dim_;
  int state_dim_;

  Eigen::VectorXd x_;
  Eigen::MatrixXd P_;
  Eigen::MatrixXd F_;
  Eigen::MatrixXd Q_;
  Eigen::MatrixXd R_;

  double q_noise_;
  double r_noise_;
  double gate_thresh_;
};

}  // namespace ibvs_filter_cpp
