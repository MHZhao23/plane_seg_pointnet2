#ifndef _planeseg_Fitting_hpp_
#define _planeseg_Fitting_hpp_

#include "Types.hpp"

namespace planeseg {

class Fitting {
public:
  enum RectangleFitAlgorithm {
    MinimumArea,
    ClosestToPriorSize,
    MaximumHullPointOverlap
  };

  struct Block {
    Eigen::Vector3f mSize;
    Eigen::Isometry3f mPose;
    std::vector<Eigen::Vector3f> mHull;
  };
  struct Result {
    bool mSuccess;
    std::vector<Block> mBlocks;
    Eigen::Vector4f mGroundPlane;
    std::vector<Eigen::Vector3f> mGroundPolygon;
  };

public:
  Fitting();

  void setBlockDimensions(const Eigen::Vector3f& iDimensions);
  void setMaxAngleOfPlaneSegmenter(const float iDegrees);
  void setMaxIterations(const int iIters);
  void setMaxEstimationError(const float iDist);
  void setRectangleFitAlgorithm(const RectangleFitAlgorithm iAlgo);
  void setDebug(const bool iVal);
  void setVisual(const bool iVal);
  void setCloud(const LabeledCloud::Ptr& iCloud);
  void setFrame(const std::string& iCloudFrame);

  Result go();

protected:
  Eigen::Vector3f mBlockDimensions;
  float mMaxAngleOfPlaneSegmenter;
  float mMaxEstimationError;
  int mMaxIterations;
  RectangleFitAlgorithm mRectangleFitAlgorithm;
  LabeledCloud::Ptr mCloud;
  std::string mCloudFrame;
  bool mDebug;
  bool mVisual;
};

}

#endif
