#include "plane_seg/PlaneFitter.hpp"

#include <limits>

#include <plane_seg/RansacGeneric.hpp>

using namespace planeseg;

namespace {

struct ProblemBase {
  MatrixX3f mPoints; // input data
  Eigen::Vector4f mPlane; // solved plane parameters

  ProblemBase(const MatrixX3f& iPoints) { mPoints = iPoints; }
  int getSampleSize() const { return 3; }
  int getNumDataPoints() const { return mPoints.rows(); }

  Eigen::Vector4f estimate(const std::vector<int>& iIndices) const {
    const int n = iIndices.size();
    Eigen::Vector4f plane;
    if (n == 3) {
      Eigen::Vector3f p1 = mPoints.row(iIndices[0]);
      Eigen::Vector3f p2 = mPoints.row(iIndices[1]);
      Eigen::Vector3f p3 = mPoints.row(iIndices[2]);
      plane.head<3>() = ((p3-p1).cross(p2-p1)).normalized();
      plane[3] = -plane.head<3>().dot(p1);
      double error = computeSumSquaredErrors(plane);
      // if (std::abs(error) > 0.02f)  plane[3] = 1e10;
    }
    else {
      MatrixX3f data(n,3);
      for (int i = 0; i < n; ++i) data.row(i) = mPoints.row(iIndices[i]);
      Eigen::Vector3f avg = data.colwise().mean();
      data.rowwise() -= avg.transpose();
      auto svd = data.jacobiSvd(Eigen::ComputeFullV);
      plane.head<3>() = svd.matrixV().col(2);
      plane[3] = -plane.head<3>().dot(avg);
    }
    return plane;
  }
  
  double computeSumSquaredErrors(const Eigen::Vector4f& iPlane) const {
    Eigen::Vector3f normal = iPlane.head<3>();
    Eigen::VectorXf errors = (mPoints*normal).array() + iPlane[3];
    double error2 = 0;
    for (int i = 0; i < (int)errors.size(); ++i) {
      error2 = error2 + errors[i]*errors[i];
    }
    // std::cout << "\nPlane error: " << error2 << ", " << std::flush;
    return error2;
  }
  
  std::vector<double> computeSquaredErrors(const Eigen::Vector4f& iPlane) const {
    Eigen::Vector3f normal = iPlane.head<3>();
    Eigen::VectorXf errors = (mPoints*normal).array() + iPlane[3];
    std::vector<double> errors2(errors.size());
    for (int i = 0; i < (int)errors2.size(); ++i) {
      errors2[i] = errors[i]*errors[i];
    }
    return errors2;
  }
};

}


PlaneFitter::
PlaneFitter() {
  // setMaxDistance(0.05);
  // setMaxIterations(100);
  setRefineUsingInliers(true);
}

PlaneFitter::
~PlaneFitter() {
}

void PlaneFitter::
setMaxDistance(const float iDistance) {
  mMaxDistance = iDistance;
}

void PlaneFitter::
setMaxIterations(const int iIterations, const float iSkipFactor) {
  mMaxIterations = iIterations;
  mSkippedIterationFactor = iSkipFactor;
}

void PlaneFitter::
setRefineUsingInliers(const bool iVal) {
  mRefineUsingInliers = iVal;
}

PlaneFitter::Result PlaneFitter::
go(const MatrixX3f& iPoints) const {
  Result result;
  drc::RansacGeneric<ProblemBase> ransac;
  ransac.setMaximumError(mMaxDistance);
  ransac.setRefineUsingInliers(mRefineUsingInliers);
  ransac.setMaximumIterations(mMaxIterations);
  ransac.setSkippedIterationFactor(mSkippedIterationFactor);

  ProblemBase problem(iPoints);

  auto res = ransac.solve(problem);
  result.mSuccess = res.mSuccess;
  result.mPlane = res.mPlane;
  result.mInliers = res.mInliers;

  return result;
}
