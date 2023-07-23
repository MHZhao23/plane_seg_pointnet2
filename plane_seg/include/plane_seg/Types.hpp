#ifndef _planeseg_Types_hpp_
#define _planeseg_Types_hpp_

//#define PCL_NO_PRECOMPILE
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp> 

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/common.h>
#include <pcl/common/angles.h> // for pcl::deg2rad
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/boundary.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/surface/concave_hull.h>
#include <thread>
#include <chrono>
#include <cmath>

namespace planeseg {

  /*
struct Point {
  PCL_ADD_POINT4D;
  float delta;
  int label;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(Point,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, delta, delta)
                                  (int, label, label)
)
  */
typedef pcl::PointXYZL Point;
typedef pcl::Normal Normal;

typedef pcl::PointCloud<pcl::Normal> NormalCloud;
typedef pcl::PointCloud<Point> LabeledCloud;
typedef Eigen::Matrix<float,Eigen::Dynamic,3> MatrixX3f;

}

#endif

