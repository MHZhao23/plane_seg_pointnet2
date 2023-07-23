#include "plane_seg/IncrementalPlaneEstimator.hpp"

#include <numeric>

using namespace planeseg;

pointsClustering::
pointsClustering() {
  reset();
}

void pointsClustering::
reset() {
  sumNormal << 0, 0, 0;
  mCount = 0;
}
// void pointsClustering::
// reset() {
//   mPoints.clear();
//   mNormals.clear();
// }

int pointsClustering::
getNumPoints() const {
  return mCount;
}
// int pointsClustering::
// getNumPoints() const {
//   return mPoints.size();
// }

void pointsClustering::
addPoint(const Eigen::Vector3f& iNormal) {
  sumNormal = sumNormal + iNormal;
  ++ mCount;
}

bool pointsClustering::
tryPoint(const Eigen::Vector3f& iNormal, const float iMaxAngle) {
  if (mCount < 2) return true;

  Eigen::Vector3f meanNormal = sumNormal.normalized();
  double angle = std::abs(pcl::getAngle3D(meanNormal, iNormal));
  if (angle > iMaxAngle) return false;

  // for (int i = 0; i < n; ++i) {
  //   double angle = std::abs(pcl::getAngle3D(mNormals[i], iNormal));
  //   if (angle > iMaxAngle*M_PI/180) return false;
  // }
  // std::cout << angle << ", " << std::flush;

  return true;
}

// void pointsClustering::
// addPoint(const Eigen::Vector3f& iPoint, const Eigen::Vector3f& iNormal) {
//   mPoints.push_back(iPoint);
//   mNormals.push_back(iNormal);
// }

// bool pointsClustering::
// tryPoint(const Eigen::Vector3f& iPoint, const Eigen::Vector3f& iNormal,
//          const float iMaxError, const float iMaxAngle) {
//   const int n = mPoints.size();
//   if (n < 2) return true;

//   Eigen::Vector3f meanNormal = Eigen::Map<Eigen::VectorXf>(mNormals.data(), mNormals.size()).sum();
//   double angle = std::abs(pcl::getAngle3D(meanNormal, iNormal));
//   if (angle > iMaxAngle*M_PI/180) return false;

//   // for (int i = 0; i < n; ++i) {
//   //   double angle = std::abs(pcl::getAngle3D(mNormals[i], iNormal));
//   //   if (angle > iMaxAngle*M_PI/180) return false;
//   // }
//   std::cout << angle << ", " << std::endl;

//   return true;
// }
