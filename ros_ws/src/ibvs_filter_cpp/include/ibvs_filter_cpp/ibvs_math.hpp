#pragma once

#include <Eigen/Dense>

namespace ibvs_filter_cpp
{

class IBVSMath
{
public:
  explicit IBVSMath(const Eigen::Matrix3d & K)
  : K_(K), K_inv_(K.inverse())
  {
  }

  Eigen::MatrixXd pixelToNormalized(const Eigen::MatrixXd & pixels) const
  {
    if (pixels.rows() != 2) {
      return Eigen::MatrixXd::Zero(2, 0);
    }

    const int n = static_cast<int>(pixels.cols());
    Eigen::MatrixXd hom_pixels(3, n);
    hom_pixels.topRows(2) = pixels;
    hom_pixels.row(2).setOnes();

    Eigen::MatrixXd norm_pixels = K_inv_ * hom_pixels;
    return norm_pixels.topRows(2);
  }

  Eigen::Matrix<double, 2, 6> getInteractionMatrixPoint(double x, double y, double z) const
  {
    Eigen::Matrix<double, 2, 6> L = Eigen::Matrix<double, 2, 6>::Zero();

    L(0, 0) = -1.0 / z;
    L(0, 1) = 0.0;
    L(0, 2) = x / z;
    L(0, 3) = x * y;
    L(0, 4) = -(1.0 + x * x);
    L(0, 5) = y;

    L(1, 0) = 0.0;
    L(1, 1) = -1.0 / z;
    L(1, 2) = y / z;
    L(1, 3) = 1.0 + y * y;
    L(1, 4) = -x * y;
    L(1, 5) = -x;

    return L;
  }

private:
  Eigen::Matrix3d K_;
  Eigen::Matrix3d K_inv_;
};

}  // namespace ibvs_filter_cpp
