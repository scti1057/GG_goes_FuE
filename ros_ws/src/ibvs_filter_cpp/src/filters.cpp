#include "ibvs_filter_cpp/filters.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ibvs_filter_cpp
{

namespace
{
bool invertible(const Eigen::MatrixXd & m)
{
  if (m.rows() != m.cols()) {
    return false;
  }
  Eigen::FullPivLU<Eigen::MatrixXd> lu(m);
  return lu.isInvertible();
}
}  // namespace

ExtendedKalmanFilter::ExtendedKalmanFilter(const Eigen::Matrix3d & K)
: BaseFilter(K),
  ibvs_math_(K),
  x_(Eigen::VectorXd::Zero(0)),
  P_(Eigen::MatrixXd::Zero(0, 0)),
  Q_(Eigen::MatrixXd::Zero(0, 0)),
  R_(Eigen::MatrixXd::Zero(0, 0)),
  q_noise_(1.0),
  r_noise_(50.0),
  gate_thresh_(20.0)
{
}

void ExtendedKalmanFilter::setStateDim(int active_count)
{
  const int l_dim = 2 * active_count;
  if (l_dim <= 0) {
    x_.resize(0);
    P_.resize(0, 0);
    Q_.resize(0, 0);
    R_.resize(0, 0);
    return;
  }
  if (x_.size() == l_dim) {
    return;
  }

  x_ = Eigen::VectorXd::Zero(l_dim);
  P_ = Eigen::MatrixXd::Identity(l_dim, l_dim) * 1000.0;
  Q_ = Eigen::MatrixXd::Identity(l_dim, l_dim) * q_noise_;
  R_ = Eigen::MatrixXd::Identity(l_dim, l_dim) * r_noise_;
}

void ExtendedKalmanFilter::setQRGate(double q_val, double r_val, double gate_thresh_val)
{
  q_noise_ = q_val;
  r_noise_ = r_val;
  gate_thresh_ = gate_thresh_val;

  const int l_dim = x_.size();
  Q_ = Eigen::MatrixXd::Identity(l_dim, l_dim) * q_noise_;
  R_ = Eigen::MatrixXd::Identity(l_dim, l_dim) * r_noise_;
}

void ExtendedKalmanFilter::forceRelocalization()
{
  BaseFilter::forceRelocalization();
  if (x_.size() > 0) {
    P_ = Eigen::MatrixXd::Identity(x_.size(), x_.size()) * 1000.0;
  }
}

Eigen::VectorXd ExtendedKalmanFilter::computePixelVelocities(
  const Eigen::VectorXd & state_2n,
  const Eigen::Matrix<double, 6, 1> & v_cam,
  double z_est) const
{
  const int l_dim = state_2n.size();
  const int n = l_dim / 2;
  Eigen::VectorXd s_dot = Eigen::VectorXd::Zero(l_dim);

  Eigen::MatrixXd pts_pixel(2, n);
  for (int i = 0; i < n; ++i) {
    pts_pixel(0, i) = state_2n(2 * i);
    pts_pixel(1, i) = state_2n(2 * i + 1);
  }

  const Eigen::MatrixXd pts_norm = ibvs_math_.pixelToNormalized(pts_pixel);

  for (int i = 0; i < n; ++i) {
    const double x_n = pts_norm(0, i);
    const double y_n = pts_norm(1, i);
    const Eigen::Matrix<double, 2, 6> L_s = ibvs_math_.getInteractionMatrixPoint(x_n, y_n, z_est);
    const Eigen::Vector2d s_dot_norm = L_s * v_cam;

    s_dot(2 * i) = s_dot_norm(0) * K_(0, 0);
    s_dot(2 * i + 1) = s_dot_norm(1) * K_(1, 1);
  }

  return s_dot;
}

void ExtendedKalmanFilter::predict(const Eigen::Matrix<double, 6, 1> & v_ee, double z_est, double dt)
{
  if (!initialized_ || x_.size() <= 0) {
    return;
  }

  const Eigen::Matrix<double, 6, 1> v_cam = transformTwistEeToCam(v_ee);
  const Eigen::VectorXd x_old = x_;
  const Eigen::VectorXd s_dot = computePixelVelocities(x_old, v_cam, z_est);

  const int l_dim = x_old.size();
  Eigen::MatrixXd F_k = Eigen::MatrixXd::Identity(l_dim, l_dim);
  constexpr double epsilon = 1e-4;

  for (int i = 0; i < l_dim; ++i) {
    Eigen::VectorXd x_plus = x_old;
    x_plus(i) += epsilon;
    const Eigen::VectorXd s_dot_plus = computePixelVelocities(x_plus, v_cam, z_est);
    const Eigen::VectorXd diff = (s_dot_plus - s_dot) / epsilon;
    F_k.col(i) += diff * dt;
  }

  x_ = x_old + s_dot * dt;
  P_ = F_k * P_ * F_k.transpose() + Q_;

  if (updateGeometryFromState(x_, false)) {
    status_ = "PREDICT";
  } else {
    status_ = "PREDICT (GEOMETRY HOLD)";
  }
}

Eigen::MatrixXd ExtendedKalmanFilter::buildObservationMatrix(const std::vector<int> & obs_slots) const
{
  const int m = static_cast<int>(obs_slots.size());
  const int l_dim = x_.size();

  Eigen::MatrixXd H_obs = Eigen::MatrixXd::Zero(2 * m, l_dim);
  for (int j = 0; j < m; ++j) {
    const int slot = obs_slots[static_cast<size_t>(j)];
    H_obs(2 * j, 2 * slot) = 1.0;
    H_obs(2 * j + 1, 2 * slot + 1) = 1.0;
  }
  return H_obs;
}

Eigen::VectorXd ExtendedKalmanFilter::getActiveStateVector() const
{
  return x_;
}

void ExtendedKalmanFilter::update(
  const Eigen::MatrixXd & current_pixels,
  const Eigen::MatrixXd & desired_pixels,
  const std::vector<int64_t> & ref_ids,
  const Eigen::VectorXd & match_scores)
{
  auto measurement = prepareMeasurement(current_pixels, desired_pixels, ref_ids, match_scores);
  if (!measurement.has_value()) {
    status_ = "REJECT (NO MATCHES)";
    return;
  }

  const Eigen::VectorXd z_k = measurement->z_k;
  const std::vector<int> & obs_slots = measurement->obs_slots;
  const std::string & prep_status = measurement->prep_status;

  setStateDim(getActiveCount());
  if (x_.size() <= 0) {
    status_ = "REJECT (NO ACTIVE)";
    return;
  }

  if (!initialized_) {
    if (z_k.size() != x_.size()) {
      status_ = "REJECT (INIT PARTIAL)";
      return;
    }

    x_ = z_k;
    initialized_ = true;
    updateGeometryFromState(x_, false);
    status_ = (prep_status == "RELOCALIZED") ? "RELOCALIZED" : "INIT";
    return;
  }

  const Eigen::MatrixXd H_obs = buildObservationMatrix(obs_slots);
  const Eigen::VectorXd y = z_k - (H_obs * x_);
  const Eigen::MatrixXd r_obs = Eigen::MatrixXd::Identity(z_k.size(), z_k.size()) * r_noise_;
  const Eigen::MatrixXd S = H_obs * P_ * H_obs.transpose() + r_obs;

  if (!invertible(S)) {
    status_ = "REJECT (SINGULAR)";
    return;
  }

  const Eigen::MatrixXd S_inv = S.inverse();
  const double mahalanobis_dist = (y.transpose() * S_inv * y)(0, 0) / std::max(1.0, static_cast<double>(z_k.size()));
  if (mahalanobis_dist > gate_thresh_) {
    status_ = "REJECT (OUTLIER)";
    updateGeometryFromState(x_, false);
    return;
  }

  const Eigen::MatrixXd K_gain = P_ * H_obs.transpose() * S_inv;
  const Eigen::VectorXd x_new = x_ + K_gain * y;

  if (updateGeometryFromState(x_new, false)) {
    x_ = x_new;
    P_ = (Eigen::MatrixXd::Identity(x_.size(), x_.size()) - K_gain * H_obs) * P_;
    status_ = "UPDATE";
  } else {
    status_ = "REJECT (GEOMETRY)";
  }
}

UnscentedKalmanFilter::UnscentedKalmanFilter(const Eigen::Matrix3d & K)
: BaseFilter(K),
  ibvs_math_(K),
  L_(0),
  x_(Eigen::VectorXd::Zero(0)),
  P_(Eigen::MatrixXd::Zero(0, 0)),
  Q_(Eigen::MatrixXd::Zero(0, 0)),
  R_(Eigen::MatrixXd::Zero(0, 0)),
  q_noise_(1.0),
  r_noise_(50.0),
  gate_thresh_(20.0),
  alpha_(1e-3),
  beta_(2.0),
  kappa_(0.0),
  lambda_(0.0),
  Wm_(Eigen::VectorXd::Zero(0)),
  Wc_(Eigen::VectorXd::Zero(0))
{
}

void UnscentedKalmanFilter::setStateDim(int active_count)
{
  const int l_new = 2 * active_count;
  if (l_new <= 0) {
    L_ = 0;
    x_.resize(0);
    P_.resize(0, 0);
    Q_.resize(0, 0);
    R_.resize(0, 0);
    Wm_.resize(0);
    Wc_.resize(0);
    return;
  }

  if (l_new == L_) {
    return;
  }

  L_ = l_new;
  x_ = Eigen::VectorXd::Zero(L_);
  P_ = Eigen::MatrixXd::Identity(L_, L_) * 1000.0;
  Q_ = Eigen::MatrixXd::Identity(L_, L_) * q_noise_;
  R_ = Eigen::MatrixXd::Identity(L_, L_) * r_noise_;
  recomputeUtWeights();
}

void UnscentedKalmanFilter::recomputeUtWeights()
{
  if (L_ <= 0) {
    lambda_ = 0.0;
    Wm_.resize(0);
    Wc_.resize(0);
    return;
  }

  lambda_ = (alpha_ * alpha_) * (L_ + kappa_) - L_;
  const int sigma_count = 2 * L_ + 1;
  Wm_ = Eigen::VectorXd::Zero(sigma_count);
  Wc_ = Eigen::VectorXd::Zero(sigma_count);

  Wm_(0) = lambda_ / (L_ + lambda_);
  Wc_(0) = Wm_(0) + (1.0 - alpha_ * alpha_ + beta_);

  for (int i = 1; i < sigma_count; ++i) {
    Wm_(i) = 1.0 / (2.0 * (L_ + lambda_));
    Wc_(i) = Wm_(i);
  }
}

void UnscentedKalmanFilter::setQRGate(double q_val, double r_val, double gate_thresh_val)
{
  q_noise_ = q_val;
  r_noise_ = r_val;
  gate_thresh_ = gate_thresh_val;

  Q_ = Eigen::MatrixXd::Identity(L_, L_) * q_noise_;
  R_ = Eigen::MatrixXd::Identity(L_, L_) * r_noise_;
}

void UnscentedKalmanFilter::forceRelocalization()
{
  BaseFilter::forceRelocalization();
  if (L_ > 0) {
    P_ = Eigen::MatrixXd::Identity(L_, L_) * 1000.0;
  }
}

Eigen::VectorXd UnscentedKalmanFilter::computePixelVelocities(
  const Eigen::VectorXd & state_2n,
  const Eigen::Matrix<double, 6, 1> & v_cam,
  double z_est) const
{
  Eigen::VectorXd s_dot = Eigen::VectorXd::Zero(L_);

  Eigen::MatrixXd pts_pixel(2, L_ / 2);
  for (int i = 0; i < (L_ / 2); ++i) {
    pts_pixel(0, i) = state_2n(2 * i);
    pts_pixel(1, i) = state_2n(2 * i + 1);
  }

  const Eigen::MatrixXd pts_norm = ibvs_math_.pixelToNormalized(pts_pixel);
  for (int i = 0; i < (L_ / 2); ++i) {
    const Eigen::Matrix<double, 2, 6> L_s = ibvs_math_.getInteractionMatrixPoint(
      pts_norm(0, i), pts_norm(1, i), z_est);
    const Eigen::Vector2d s_dot_norm = L_s * v_cam;

    s_dot(2 * i) = s_dot_norm(0) * K_(0, 0);
    s_dot(2 * i + 1) = s_dot_norm(1) * K_(1, 1);
  }

  return s_dot;
}

Eigen::MatrixXd UnscentedKalmanFilter::generateSigmaPoints(const Eigen::VectorXd & x, Eigen::MatrixXd P)
{
  P = (P + P.transpose()) * 0.5;
  P += Eigen::MatrixXd::Identity(L_, L_) * 1e-8;

  Eigen::LLT<Eigen::MatrixXd> llt(P);
  Eigen::MatrixXd L_chol;
  if (llt.info() == Eigen::Success) {
    L_chol = llt.matrixL();
  } else {
    L_chol = Eigen::MatrixXd::Identity(L_, L_) * 0.1;
    P_ = Eigen::MatrixXd::Identity(L_, L_) * 10.0;
  }

  Eigen::MatrixXd sigma_points = Eigen::MatrixXd::Zero(L_, 2 * L_ + 1);
  sigma_points.col(0) = x;

  const double gamma = std::sqrt(L_ + lambda_);
  for (int i = 0; i < L_; ++i) {
    sigma_points.col(i + 1) = x + gamma * L_chol.col(i);
    sigma_points.col(L_ + i + 1) = x - gamma * L_chol.col(i);
  }

  return sigma_points;
}

void UnscentedKalmanFilter::predict(const Eigen::Matrix<double, 6, 1> & v_ee, double z_est, double dt)
{
  if (!initialized_ || L_ <= 0) {
    return;
  }

  const Eigen::Matrix<double, 6, 1> v_cam = transformTwistEeToCam(v_ee);

  const Eigen::MatrixXd sigmas = generateSigmaPoints(x_, P_);
  Eigen::MatrixXd sigmas_pred = Eigen::MatrixXd::Zero(sigmas.rows(), sigmas.cols());

  for (int i = 0; i < sigmas.cols(); ++i) {
    const Eigen::VectorXd x_i = sigmas.col(i);
    const Eigen::VectorXd s_dot_i = computePixelVelocities(x_i, v_cam, z_est);
    sigmas_pred.col(i) = x_i + s_dot_i * dt;
  }

  Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(L_);
  for (int i = 0; i < sigmas_pred.cols(); ++i) {
    x_pred += Wm_(i) * sigmas_pred.col(i);
  }

  Eigen::MatrixXd P_pred = Eigen::MatrixXd::Zero(L_, L_);
  for (int i = 0; i < sigmas_pred.cols(); ++i) {
    const Eigen::VectorXd y = sigmas_pred.col(i) - x_pred;
    P_pred += Wc_(i) * (y * y.transpose());
  }

  x_ = x_pred;
  P_ = P_pred + Q_;

  if (updateGeometryFromState(x_, false)) {
    status_ = "PREDICT";
  } else {
    status_ = "PREDICT (GEOMETRY HOLD)";
  }
}

Eigen::MatrixXd UnscentedKalmanFilter::buildObservationMatrix(const std::vector<int> & obs_slots) const
{
  const int m = static_cast<int>(obs_slots.size());
  Eigen::MatrixXd H_obs = Eigen::MatrixXd::Zero(2 * m, L_);

  for (int j = 0; j < m; ++j) {
    const int slot = obs_slots[static_cast<size_t>(j)];
    H_obs(2 * j, 2 * slot) = 1.0;
    H_obs(2 * j + 1, 2 * slot + 1) = 1.0;
  }

  return H_obs;
}

Eigen::VectorXd UnscentedKalmanFilter::getActiveStateVector() const
{
  return x_;
}

void UnscentedKalmanFilter::update(
  const Eigen::MatrixXd & current_pixels,
  const Eigen::MatrixXd & desired_pixels,
  const std::vector<int64_t> & ref_ids,
  const Eigen::VectorXd & match_scores)
{
  auto measurement = prepareMeasurement(current_pixels, desired_pixels, ref_ids, match_scores);
  if (!measurement.has_value()) {
    status_ = "REJECT (NO MATCHES)";
    return;
  }

  const Eigen::VectorXd z_k = measurement->z_k;
  const std::vector<int> & obs_slots = measurement->obs_slots;
  const std::string & prep_status = measurement->prep_status;

  setStateDim(getActiveCount());
  if (L_ <= 0) {
    status_ = "REJECT (NO ACTIVE)";
    return;
  }

  if (!initialized_) {
    if (z_k.size() != L_) {
      status_ = "REJECT (INIT PARTIAL)";
      return;
    }

    x_ = z_k;
    initialized_ = true;
    updateGeometryFromState(x_, false);
    status_ = (prep_status == "RELOCALIZED") ? "RELOCALIZED" : "INIT";
    return;
  }

  const Eigen::MatrixXd H_obs = buildObservationMatrix(obs_slots);
  const Eigen::VectorXd y = z_k - (H_obs * x_);
  const Eigen::MatrixXd r_obs = Eigen::MatrixXd::Identity(z_k.size(), z_k.size()) * r_noise_;
  const Eigen::MatrixXd S = H_obs * P_ * H_obs.transpose() + r_obs;

  if (!invertible(S)) {
    status_ = "REJECT (SINGULAR)";
    return;
  }

  const Eigen::MatrixXd S_inv = S.inverse();
  const double nis = (y.transpose() * S_inv * y)(0, 0) / std::max(1.0, static_cast<double>(z_k.size()));
  if (nis > gate_thresh_) {
    status_ = "REJECT (OUTLIER)";
    updateGeometryFromState(x_, false);
    return;
  }

  const Eigen::MatrixXd K_gain = P_ * H_obs.transpose() * S_inv;
  const Eigen::VectorXd x_new = x_ + K_gain * y;

  if (updateGeometryFromState(x_new, false)) {
    x_ = x_new;
    P_ = (Eigen::MatrixXd::Identity(L_, L_) - K_gain * H_obs) * P_;
    status_ = "UPDATE";
  } else {
    status_ = "REJECT (GEOMETRY)";
  }
}

ErrorStateKalmanFilter::ErrorStateKalmanFilter(const Eigen::Matrix3d & K)
: BaseFilter(K),
  ibvs_math_(K),
  L_(0),
  x_nom_(Eigen::VectorXd::Zero(0)),
  dx_(Eigen::VectorXd::Zero(0)),
  P_(Eigen::MatrixXd::Zero(0, 0)),
  Q_(Eigen::MatrixXd::Zero(0, 0)),
  R_(Eigen::MatrixXd::Zero(0, 0)),
  q_noise_(1.0),
  r_noise_(50.0),
  gate_thresh_(20.0)
{
}

void ErrorStateKalmanFilter::setStateDim(int active_count)
{
  const int l_new = 2 * active_count;
  if (l_new <= 0) {
    L_ = 0;
    x_nom_.resize(0);
    dx_.resize(0);
    P_.resize(0, 0);
    Q_.resize(0, 0);
    R_.resize(0, 0);
    return;
  }

  if (l_new == L_) {
    return;
  }

  L_ = l_new;
  x_nom_ = Eigen::VectorXd::Zero(L_);
  dx_ = Eigen::VectorXd::Zero(L_);
  P_ = Eigen::MatrixXd::Identity(L_, L_) * 1000.0;
  Q_ = Eigen::MatrixXd::Identity(L_, L_) * q_noise_;
  R_ = Eigen::MatrixXd::Identity(L_, L_) * r_noise_;
}

void ErrorStateKalmanFilter::setQRGate(double q_val, double r_val, double gate_thresh_val)
{
  q_noise_ = q_val;
  r_noise_ = r_val;
  gate_thresh_ = gate_thresh_val;

  Q_ = Eigen::MatrixXd::Identity(L_, L_) * q_noise_;
  R_ = Eigen::MatrixXd::Identity(L_, L_) * r_noise_;
}

void ErrorStateKalmanFilter::forceRelocalization()
{
  BaseFilter::forceRelocalization();
  if (L_ > 0) {
    P_ = Eigen::MatrixXd::Identity(L_, L_) * 1000.0;
    dx_ = Eigen::VectorXd::Zero(L_);
  }
}

Eigen::VectorXd ErrorStateKalmanFilter::computePixelVelocities(
  const Eigen::VectorXd & state_2n,
  const Eigen::Matrix<double, 6, 1> & v_cam,
  double z_est) const
{
  Eigen::VectorXd s_dot = Eigen::VectorXd::Zero(L_);

  Eigen::MatrixXd pts_pixel(2, L_ / 2);
  for (int i = 0; i < (L_ / 2); ++i) {
    pts_pixel(0, i) = state_2n(2 * i);
    pts_pixel(1, i) = state_2n(2 * i + 1);
  }

  const Eigen::MatrixXd pts_norm = ibvs_math_.pixelToNormalized(pts_pixel);
  for (int i = 0; i < (L_ / 2); ++i) {
    const Eigen::Matrix<double, 2, 6> L_s = ibvs_math_.getInteractionMatrixPoint(
      pts_norm(0, i), pts_norm(1, i), z_est);
    const Eigen::Vector2d s_dot_norm = L_s * v_cam;

    s_dot(2 * i) = s_dot_norm(0) * K_(0, 0);
    s_dot(2 * i + 1) = s_dot_norm(1) * K_(1, 1);
  }

  return s_dot;
}

void ErrorStateKalmanFilter::predict(
  const Eigen::Matrix<double, 6, 1> & v_ee,
  double z_est,
  double dt)
{
  if (!initialized_ || L_ <= 0) {
    return;
  }

  const Eigen::Matrix<double, 6, 1> v_cam = transformTwistEeToCam(v_ee);

  const Eigen::VectorXd x_nom_old = x_nom_;
  const Eigen::VectorXd s_dot = computePixelVelocities(x_nom_old, v_cam, z_est);
  x_nom_ = x_nom_old + s_dot * dt;

  Eigen::MatrixXd F_dx = Eigen::MatrixXd::Identity(L_, L_);
  constexpr double epsilon = 1e-4;
  for (int i = 0; i < L_; ++i) {
    Eigen::VectorXd x_plus = x_nom_old;
    x_plus(i) += epsilon;
    const Eigen::VectorXd s_dot_plus = computePixelVelocities(x_plus, v_cam, z_est);
    F_dx.col(i) += ((s_dot_plus - s_dot) / epsilon) * dt;
  }

  P_ = F_dx * P_ * F_dx.transpose() + Q_;

  if (updateGeometryFromState(x_nom_, false)) {
    status_ = "PREDICT";
  } else {
    status_ = "PREDICT (GEOMETRY HOLD)";
  }
}

Eigen::MatrixXd ErrorStateKalmanFilter::buildObservationMatrix(const std::vector<int> & obs_slots) const
{
  const int m = static_cast<int>(obs_slots.size());
  Eigen::MatrixXd H_obs = Eigen::MatrixXd::Zero(2 * m, L_);

  for (int j = 0; j < m; ++j) {
    const int slot = obs_slots[static_cast<size_t>(j)];
    H_obs(2 * j, 2 * slot) = 1.0;
    H_obs(2 * j + 1, 2 * slot + 1) = 1.0;
  }

  return H_obs;
}

Eigen::VectorXd ErrorStateKalmanFilter::getActiveStateVector() const
{
  return x_nom_;
}

void ErrorStateKalmanFilter::update(
  const Eigen::MatrixXd & current_pixels,
  const Eigen::MatrixXd & desired_pixels,
  const std::vector<int64_t> & ref_ids,
  const Eigen::VectorXd & match_scores)
{
  auto measurement = prepareMeasurement(current_pixels, desired_pixels, ref_ids, match_scores);
  if (!measurement.has_value()) {
    status_ = "REJECT (NO MATCHES)";
    return;
  }

  const Eigen::VectorXd z_k = measurement->z_k;
  const std::vector<int> & obs_slots = measurement->obs_slots;
  const std::string & prep_status = measurement->prep_status;

  setStateDim(getActiveCount());
  if (L_ <= 0) {
    status_ = "REJECT (NO ACTIVE)";
    return;
  }

  if (!initialized_) {
    if (z_k.size() != L_) {
      status_ = "REJECT (INIT PARTIAL)";
      return;
    }

    x_nom_ = z_k;
    dx_ = Eigen::VectorXd::Zero(L_);
    initialized_ = true;
    updateGeometryFromState(x_nom_, false);
    status_ = (prep_status == "RELOCALIZED") ? "RELOCALIZED" : "INIT";
    return;
  }

  const Eigen::MatrixXd H_obs = buildObservationMatrix(obs_slots);
  const Eigen::VectorXd y = z_k - (H_obs * x_nom_);
  const Eigen::MatrixXd r_obs = Eigen::MatrixXd::Identity(z_k.size(), z_k.size()) * r_noise_;
  const Eigen::MatrixXd S = H_obs * P_ * H_obs.transpose() + r_obs;

  if (!invertible(S)) {
    status_ = "REJECT (SINGULAR)";
    return;
  }

  const Eigen::MatrixXd S_inv = S.inverse();
  const double nis = (y.transpose() * S_inv * y)(0, 0) / std::max(1.0, static_cast<double>(z_k.size()));
  if (nis > gate_thresh_) {
    status_ = "REJECT (OUTLIER)";
    updateGeometryFromState(x_nom_, false);
    return;
  }

  const Eigen::MatrixXd K_gain = P_ * H_obs.transpose() * S_inv;
  dx_ = K_gain * y;
  P_ = (Eigen::MatrixXd::Identity(L_, L_) - K_gain * H_obs) * P_;

  const Eigen::VectorXd x_new = x_nom_ + dx_;
  if (updateGeometryFromState(x_new, false)) {
    x_nom_ = x_new;
    dx_ = Eigen::VectorXd::Zero(L_);
    status_ = "UPDATE";
  } else {
    status_ = "REJECT (GEOMETRY)";
  }
}

StandardKalmanFilter::StandardKalmanFilter(const Eigen::Matrix3d & K)
: BaseFilter(K),
  keypoint_count_(0),
  pos_dim_(0),
  state_dim_(0),
  x_(Eigen::VectorXd::Zero(0)),
  P_(Eigen::MatrixXd::Zero(0, 0)),
  F_(Eigen::MatrixXd::Zero(0, 0)),
  Q_(Eigen::MatrixXd::Zero(0, 0)),
  R_(Eigen::MatrixXd::Zero(0, 0)),
  q_noise_(100.0),
  r_noise_(50.0),
  gate_thresh_(20.0)
{
}

void StandardKalmanFilter::setStateDim(int active_count)
{
  const int n = active_count;
  if (n <= 0) {
    keypoint_count_ = 0;
    pos_dim_ = 0;
    state_dim_ = 0;
    x_.resize(0);
    P_.resize(0, 0);
    F_.resize(0, 0);
    Q_.resize(0, 0);
    R_.resize(0, 0);
    return;
  }

  if (n == keypoint_count_) {
    return;
  }

  keypoint_count_ = n;
  pos_dim_ = 2 * n;
  state_dim_ = 4 * n;

  x_ = Eigen::VectorXd::Zero(state_dim_);
  P_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_) * 1000.0;
  F_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_);

  Q_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_) * (q_noise_ * 0.1);
  if (state_dim_ > pos_dim_) {
    Q_.block(pos_dim_, pos_dim_, state_dim_ - pos_dim_, state_dim_ - pos_dim_) =
      Eigen::MatrixXd::Identity(state_dim_ - pos_dim_, state_dim_ - pos_dim_) * q_noise_;
  }

  R_ = Eigen::MatrixXd::Identity(pos_dim_, pos_dim_) * r_noise_;
}

void StandardKalmanFilter::setQRGate(double q_val, double r_val, double gate_thresh_val)
{
  q_noise_ = q_val;
  r_noise_ = r_val;
  gate_thresh_ = gate_thresh_val;

  Q_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_) * (q_noise_ * 0.1);
  if (state_dim_ > pos_dim_) {
    Q_.block(pos_dim_, pos_dim_, state_dim_ - pos_dim_, state_dim_ - pos_dim_) =
      Eigen::MatrixXd::Identity(state_dim_ - pos_dim_, state_dim_ - pos_dim_) * q_noise_;
  }
  R_ = Eigen::MatrixXd::Identity(pos_dim_, pos_dim_) * r_noise_;
}

void StandardKalmanFilter::forceRelocalization()
{
  BaseFilter::forceRelocalization();
  if (state_dim_ > 0) {
    P_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_) * 1000.0;
  }
}

void StandardKalmanFilter::predict(
  const Eigen::Matrix<double, 6, 1> & /*v_ee*/,
  double /*z_est*/,
  double dt)
{
  if (!initialized_ || state_dim_ <= 0) {
    return;
  }

  F_ = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
  for (int i = 0; i < pos_dim_; ++i) {
    F_(i, i + pos_dim_) = dt;
  }

  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;

  if (updateGeometryFromState(x_.head(pos_dim_), false)) {
    status_ = "PREDICT";
  } else {
    status_ = "PREDICT (GEOMETRY HOLD)";
  }
}

Eigen::MatrixXd StandardKalmanFilter::buildObservationMatrix(const std::vector<int> & obs_slots) const
{
  const int m = static_cast<int>(obs_slots.size());
  Eigen::MatrixXd H_obs = Eigen::MatrixXd::Zero(2 * m, state_dim_);

  for (int j = 0; j < m; ++j) {
    const int slot = obs_slots[static_cast<size_t>(j)];
    H_obs(2 * j, 2 * slot) = 1.0;
    H_obs(2 * j + 1, 2 * slot + 1) = 1.0;
  }

  return H_obs;
}

Eigen::VectorXd StandardKalmanFilter::getActiveStateVector() const
{
  return x_.head(pos_dim_);
}

void StandardKalmanFilter::update(
  const Eigen::MatrixXd & current_pixels,
  const Eigen::MatrixXd & desired_pixels,
  const std::vector<int64_t> & ref_ids,
  const Eigen::VectorXd & match_scores)
{
  auto measurement = prepareMeasurement(current_pixels, desired_pixels, ref_ids, match_scores);
  if (!measurement.has_value()) {
    status_ = "REJECT (NO MATCHES)";
    return;
  }

  const Eigen::VectorXd z_k = measurement->z_k;
  const std::vector<int> & obs_slots = measurement->obs_slots;
  const std::string & prep_status = measurement->prep_status;

  setStateDim(getActiveCount());
  if (state_dim_ <= 0) {
    status_ = "REJECT (NO ACTIVE)";
    return;
  }

  if (!initialized_) {
    if (z_k.size() != pos_dim_) {
      status_ = "REJECT (INIT PARTIAL)";
      return;
    }

    x_.head(pos_dim_) = z_k;
    x_.tail(state_dim_ - pos_dim_).setZero();
    initialized_ = true;
    updateGeometryFromState(x_.head(pos_dim_), false);
    status_ = (prep_status == "RELOCALIZED") ? "RELOCALIZED" : "INIT";
    return;
  }

  const Eigen::MatrixXd H_obs = buildObservationMatrix(obs_slots);
  const Eigen::VectorXd y = z_k - (H_obs * x_);
  const Eigen::MatrixXd r_obs = Eigen::MatrixXd::Identity(z_k.size(), z_k.size()) * r_noise_;
  const Eigen::MatrixXd S = H_obs * P_ * H_obs.transpose() + r_obs;

  if (!invertible(S)) {
    status_ = "REJECT (SINGULAR)";
    return;
  }

  const Eigen::MatrixXd S_inv = S.inverse();
  const double mahalanobis_dist = (y.transpose() * S_inv * y)(0, 0) / std::max(1.0, static_cast<double>(z_k.size()));
  if (mahalanobis_dist > gate_thresh_) {
    status_ = "REJECT (OUTLIER)";
    updateGeometryFromState(x_.head(pos_dim_), false);
    return;
  }

  const Eigen::MatrixXd K_gain = P_ * H_obs.transpose() * S_inv;
  const Eigen::VectorXd x_new = x_ + K_gain * y;

  if (updateGeometryFromState(x_new.head(pos_dim_), false)) {
    x_ = x_new;
    P_ = (Eigen::MatrixXd::Identity(state_dim_, state_dim_) - K_gain * H_obs) * P_;
    status_ = "UPDATE";
  } else {
    status_ = "REJECT (GEOMETRY)";
  }
}

}  // namespace ibvs_filter_cpp
