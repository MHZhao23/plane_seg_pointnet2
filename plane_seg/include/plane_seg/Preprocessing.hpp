#ifndef _planeseg_Preprocessing_hpp_
#define _planeseg_Preprocessing_hpp_

#include "Types.hpp"

namespace planeseg {

class Preprocessing {
public:
    Preprocessing();

    void setSensorPose(const Eigen::Vector3f& iOrigin,
                        const Eigen::Vector3f& iLookDir);
    void setDownsampleResolution(const float iRes);
    void setMaxAngleFromHorizontal(const float iDegrees);
    void setCloud(const LabeledCloud::Ptr& iCloud);
    void setFrame(const std::string& iCloudFrame);
    void setDebug(const bool iVal);
    void setVisual(const bool iVal);
    LabeledCloud::Ptr go();

protected:
    Eigen::Vector3f mOrigin;
    Eigen::Vector3f mLookDir;
    float mDownsampleResolution;
    float mMaxAngleFromHorizontal;
    LabeledCloud::Ptr mCloud;
    std::string mCloudFrame;
    bool mDebug;
    bool mVisual;
};

}

#endif
