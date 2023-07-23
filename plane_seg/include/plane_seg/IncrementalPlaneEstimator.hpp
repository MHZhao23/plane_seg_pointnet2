#ifndef _planeseg_IncrementalPlaneEstimator_hpp_
#define _planeseg_IncrementalPlaneEstimator_hpp_

#include <vector>
#include "Types.hpp"

namespace planeseg {

class pointsClustering {
protected:
  // std::vector<Eigen::Vector3f> mPoints;
  // std::vector<Eigen::Vector3f> mNormals;
  Eigen::Vector3f sumNormal;
  int mCount;

public:
  pointsClustering();

  void reset();
  int getNumPoints() const;
  void addPoint(const Eigen::Vector3f& iNormal);
  bool tryPoint(const Eigen::Vector3f& iNormal, const float iMaxAngle);
  // void addPoint(const Eigen::Vector3f& iPoint, const Eigen::Vector3f& iNormal);
  // bool tryPoint(const Eigen::Vector3f& iPoint, const Eigen::Vector3f& iNormal,
  //               const float iMaxError, const float iMaxAngle);
};

}

#endif
