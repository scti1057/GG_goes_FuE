#include "ibvs_filter_cpp/base_filter.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace ibvs_filter_cpp
{

BaseFilter::BaseFilter(const Eigen::Matrix3d & K)
: K_(K),
  initialized_(false),
  status_("INIT"),
  H_filtered_(Eigen::Matrix3d::Identity()),
  active_ref_ids_(),
  active_desired_(Eigen::MatrixXd::Zero(2, 0)),
  max_active_keypoints_(80),
  min_init_keypoints_(8),
  min_reinit_unique_matches_(8),
  min_update_keypoints_(4)
{
}

void BaseFilter::reset()
{
  initialized_ = false;
  status_ = "INIT";
  H_filtered_ = Eigen::Matrix3d::Identity();
  active_ref_ids_.clear();
  active_desired_.resize(2, 0);
}

void BaseFilter::forceRelocalization()
{
  initialized_ = false;
  status_ = "RELOCALIZING";
  active_ref_ids_.clear();
  active_desired_.resize(2, 0);
  H_filtered_ = Eigen::Matrix3d::Identity();
}

void BaseFilter::configureKeypointTracking(
  int max_active_keypoints,
  int min_init_keypoints,
  int min_update_keypoints)
{
  max_active_keypoints_ = std::max(4, max_active_keypoints);
  min_reinit_unique_matches_ = std::max(4, min_init_keypoints);
  // Keep startup behavior stable: initial model build still uses a threshold
  // clamped by active-set size, while reinit can require more unique matches.
  min_init_keypoints_ = std::min(min_reinit_unique_matches_, max_active_keypoints_);
  min_update_keypoints_ = std::max(0, min_update_keypoints);

  min_update_keypoints_ = std::min(min_update_keypoints_, max_active_keypoints_);
}

int BaseFilter::getActiveCount() const
{
  return static_cast<int>(active_ref_ids_.size());
}

std::vector<int64_t> BaseFilter::getActiveRefIds() const
{
  return active_ref_ids_;
}

Eigen::VectorXd BaseFilter::stackMeasurement(const Eigen::MatrixXd & current_obs)
{
  const int m = static_cast<int>(current_obs.cols());
  Eigen::VectorXd z = Eigen::VectorXd::Zero(2 * m);
  for (int i = 0; i < m; ++i) {
    z(2 * i) = current_obs(0, i);
    z(2 * i + 1) = current_obs(1, i);
  }
  return z;
}

std::vector<int> BaseFilter::selectSpreadIndices(
  const Eigen::MatrixXd & desired_pixels,
  const Eigen::VectorXd & match_scores,
  int max_count) const
{
  const int m = static_cast<int>(desired_pixels.cols());
  if (m <= max_count) {
    std::vector<int> all(m);
    for (int i = 0; i < m; ++i) {
      all[i] = i;
    }
    return all;
  }

  Eigen::VectorXd scores = Eigen::VectorXd::Zero(m);
  if (match_scores.size() == m) {
    scores = match_scores;
    bool all_finite = true;
    for (int i = 0; i < m; ++i) {
      if (!std::isfinite(scores(i))) {
        all_finite = false;
        break;
      }
    }
    if (all_finite) {
      const double s_min = scores.minCoeff();
      const double s_max = scores.maxCoeff();
      if (s_max > s_min) {
        scores = (scores.array() - s_min) / (s_max - s_min);
      } else {
        scores.setZero();
      }
    } else {
      scores.setZero();
    }
  }

  const Eigen::Vector2d centroid = desired_pixels.rowwise().mean();
  Eigen::VectorXd dist_to_centroid = Eigen::VectorXd::Zero(m);
  for (int i = 0; i < m; ++i) {
    const Eigen::Vector2d d = desired_pixels.col(i) - centroid;
    dist_to_centroid(i) = d.squaredNorm();
  }

  int first_idx = 0;
  if (scores.maxCoeff() > 0.0) {
    scores.maxCoeff(&first_idx);
  } else {
    dist_to_centroid.maxCoeff(&first_idx);
  }

  std::vector<int> selected;
  selected.reserve(max_count);
  selected.push_back(first_idx);

  Eigen::VectorXd min_dist2 = Eigen::VectorXd::Constant(m, std::numeric_limits<double>::infinity());
  constexpr double score_weight = 0.05;

  for (int k = 1; k < max_count; ++k) {
    const int last_idx = selected.back();

    for (int i = 0; i < m; ++i) {
      const Eigen::Vector2d d = desired_pixels.col(i) - desired_pixels.col(last_idx);
      min_dist2(i) = std::min(min_dist2(i), d.squaredNorm());
    }

    double finite_max = 1.0;
    for (int i = 0; i < m; ++i) {
      if (std::isfinite(min_dist2(i))) {
        finite_max = std::max(finite_max, min_dist2(i));
      }
    }

    Eigen::VectorXd objective = min_dist2 + score_weight * finite_max * scores;
    for (const int idx : selected) {
      objective(idx) = -std::numeric_limits<double>::infinity();
    }

    int next_idx = 0;
    objective.maxCoeff(&next_idx);
    if (!std::isfinite(objective(next_idx))) {
      break;
    }
    selected.push_back(next_idx);
  }

  return selected;
}

std::optional<Eigen::MatrixXd> BaseFilter::initializeActiveSet(
  const Eigen::MatrixXd & current_pixels,
  const Eigen::MatrixXd & desired_pixels,
  const std::vector<int64_t> & ref_ids,
  const Eigen::VectorXd & match_scores,
  bool strict_reinit_threshold)
{
  int m = static_cast<int>(desired_pixels.cols());
  if (m <= 0) {
    return std::nullopt;
  }

  const bool has_scores = (match_scores.size() == m);

  std::map<int64_t, std::pair<double, int>> best_per_id;
  for (int i = 0; i < m; ++i) {
    const int64_t rid = ref_ids[static_cast<size_t>(i)];
    const double sc = has_scores ? match_scores(i) : 0.0;
    auto it = best_per_id.find(rid);
    if (it == best_per_id.end() || sc > it->second.first) {
      best_per_id[rid] = {sc, i};
    }
  }

  std::vector<int> unique_indices;
  unique_indices.reserve(best_per_id.size());
  for (const auto & kv : best_per_id) {
    unique_indices.push_back(kv.second.second);
  }
  std::sort(unique_indices.begin(), unique_indices.end());

  Eigen::MatrixXd current_unique(2, static_cast<int>(unique_indices.size()));
  Eigen::MatrixXd desired_unique(2, static_cast<int>(unique_indices.size()));
  std::vector<int64_t> ref_ids_unique(unique_indices.size(), 0);
  Eigen::VectorXd scores_unique = Eigen::VectorXd::Zero(static_cast<int>(unique_indices.size()));

  for (size_t i = 0; i < unique_indices.size(); ++i) {
    const int idx = unique_indices[i];
    current_unique.col(static_cast<int>(i)) = current_pixels.col(idx);
    desired_unique.col(static_cast<int>(i)) = desired_pixels.col(idx);
    ref_ids_unique[i] = ref_ids[static_cast<size_t>(idx)];
    if (has_scores) {
      scores_unique(static_cast<int>(i)) = match_scores(idx);
    }
  }

  m = static_cast<int>(desired_unique.cols());
  const int required_unique =
    strict_reinit_threshold ? min_reinit_unique_matches_ : min_init_keypoints_;
  if (m < required_unique) {
    return std::nullopt;
  }

  const auto selected = selectSpreadIndices(
    desired_unique,
    scores_unique,
    std::min(max_active_keypoints_, m));

  active_ref_ids_.clear();
  active_ref_ids_.reserve(selected.size());
  active_desired_.resize(2, static_cast<int>(selected.size()));

  Eigen::MatrixXd current_selected(2, static_cast<int>(selected.size()));
  for (size_t i = 0; i < selected.size(); ++i) {
    const int idx = selected[i];
    active_ref_ids_.push_back(ref_ids_unique[static_cast<size_t>(idx)]);
    active_desired_.col(static_cast<int>(i)) = desired_unique.col(idx);
    current_selected.col(static_cast<int>(i)) = current_unique.col(idx);
  }

  initialized_ = false;
  H_filtered_ = Eigen::Matrix3d::Identity();
  return current_selected;
}

std::optional<MeasurementData> BaseFilter::prepareMeasurement(
  const Eigen::MatrixXd & current_pixels,
  const Eigen::MatrixXd & desired_pixels,
  const std::vector<int64_t> & ref_ids,
  const Eigen::VectorXd & match_scores)
{
  if (current_pixels.rows() != 2 || desired_pixels.rows() != 2) {
    return std::nullopt;
  }

  const int m = std::min(current_pixels.cols(), desired_pixels.cols());
  if (m <= 0) {
    return std::nullopt;
  }

  Eigen::MatrixXd current = current_pixels.leftCols(m);
  Eigen::MatrixXd desired = desired_pixels.leftCols(m);

  std::vector<int64_t> ids;
  ids.reserve(static_cast<size_t>(m));
  for (int i = 0; i < m; ++i) {
    if (i < static_cast<int>(ref_ids.size())) {
      ids.push_back(ref_ids[static_cast<size_t>(i)]);
    } else {
      ids.push_back(static_cast<int64_t>(i));
    }
  }

  Eigen::VectorXd scores = Eigen::VectorXd::Zero(m);
  if (match_scores.size() >= m) {
    scores = match_scores.head(m);
  }

  if (active_ref_ids_.empty()) {
    // Initial startup: keep legacy/tolerant init threshold behavior.
    auto current_selected = initializeActiveSet(current, desired, ids, scores, false);
    if (!current_selected.has_value()) {
      return std::nullopt;
    }

    MeasurementData out;
    out.z_k = stackMeasurement(current_selected.value());
    out.prep_status = "INIT SET";
    out.obs_slots.resize(active_ref_ids_.size());
    for (size_t i = 0; i < active_ref_ids_.size(); ++i) {
      out.obs_slots[i] = static_cast<int>(i);
    }
    return out;
  }

  std::map<int64_t, int> id_to_slot;
  for (size_t i = 0; i < active_ref_ids_.size(); ++i) {
    id_to_slot[active_ref_ids_[i]] = static_cast<int>(i);
  }

  std::vector<int> obs_slots;
  std::vector<Eigen::Vector2d> obs_points;
  std::vector<bool> seen(static_cast<size_t>(active_ref_ids_.size()), false);

  for (int i = 0; i < m; ++i) {
    const auto it = id_to_slot.find(ids[static_cast<size_t>(i)]);
    if (it == id_to_slot.end()) {
      continue;
    }
    const int slot = it->second;
    if (seen[static_cast<size_t>(slot)]) {
      continue;
    }
    seen[static_cast<size_t>(slot)] = true;
    obs_slots.push_back(slot);
    obs_points.push_back(current.col(i));
  }

  if (static_cast<int>(obs_slots.size()) < min_update_keypoints_) {
    // Reinit after tracking loss: require strict unique-match threshold.
    auto current_selected = initializeActiveSet(current, desired, ids, scores, true);
    if (!current_selected.has_value()) {
      return std::nullopt;
    }

    MeasurementData out;
    out.z_k = stackMeasurement(current_selected.value());
    out.prep_status = "RELOCALIZED";
    out.obs_slots.resize(active_ref_ids_.size());
    for (size_t i = 0; i < active_ref_ids_.size(); ++i) {
      out.obs_slots[i] = static_cast<int>(i);
    }
    return out;
  }

  Eigen::MatrixXd current_obs(2, static_cast<int>(obs_points.size()));
  for (size_t i = 0; i < obs_points.size(); ++i) {
    current_obs.col(static_cast<int>(i)) = obs_points[i];
  }

  MeasurementData out;
  out.z_k = stackMeasurement(current_obs);
  out.obs_slots = std::move(obs_slots);
  out.prep_status = status_;
  return out;
}

bool BaseFilter::checkGeometryAndUpdateH(const Eigen::VectorXd & state_2n, bool log_on_fail)
{
  const int n = getActiveCount();
  if (n < 4) {
    return false;
  }

  if (state_2n.size() != (2 * n)) {
    return false;
  }

  cv::Mat pts_cur(n, 2, CV_32F);
  cv::Mat pts_des(n, 2, CV_32F);

  for (int i = 0; i < n; ++i) {
    pts_cur.at<float>(i, 0) = static_cast<float>(state_2n(2 * i));
    pts_cur.at<float>(i, 1) = static_cast<float>(state_2n(2 * i + 1));
    pts_des.at<float>(i, 0) = static_cast<float>(active_desired_(0, i));
    pts_des.at<float>(i, 1) = static_cast<float>(active_desired_(1, i));
  }

  cv::Mat H = cv::findHomography(pts_des, pts_cur, 0);
  if (H.empty()) {
    (void)log_on_fail;
    return false;
  }

  Eigen::Matrix3d H_eig = Eigen::Matrix3d::Identity();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      const double v = H.at<double>(r, c);
      if (!std::isfinite(v)) {
        return false;
      }
      H_eig(r, c) = v;
    }
  }

  if (std::abs(H_eig(2, 2)) > 1e-12) {
    H_eig /= H_eig(2, 2);
  }

  H_filtered_ = H_eig;
  return true;
}

bool BaseFilter::updateGeometryFromState(const Eigen::VectorXd & state_2n, bool log_on_fail)
{
  if (getActiveCount() < 4) {
    return false;
  }
  return checkGeometryAndUpdateH(state_2n, log_on_fail);
}

Eigen::MatrixXd BaseFilter::getActiveFilteredPoints() const
{
  const int n = getActiveCount();
  if (!initialized_ || n <= 0) {
    return Eigen::MatrixXd::Zero(2, 0);
  }

  Eigen::VectorXd state = getActiveStateVector();
  if (state.size() != (2 * n)) {
    return Eigen::MatrixXd::Zero(2, 0);
  }

  Eigen::MatrixXd points(2, n);
  for (int i = 0; i < n; ++i) {
    points(0, i) = state(2 * i);
    points(1, i) = state(2 * i + 1);
  }
  return points;
}

}  // namespace ibvs_filter_cpp
