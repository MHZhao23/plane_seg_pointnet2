#include "plane_seg/BlockFitter.hpp"

#include <chrono>
#include <fstream>

// PCL's octree_key.h (included from convex_hull.h) uses anonymous structs and nested anonymous unions.
// These are GNU extensions - we want to ignore warnings about them, though.

#if defined(__clang__)
# pragma clang diagnostic push
#endif

#if defined(__clang__) && defined(__has_warning)
# if __has_warning( "-Wgnu-anonymous-struct" )
#  pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
# endif
# if __has_warning( "-Wnested-anon-types" )
#  pragma clang diagnostic ignored "-Wnested-anon-types"
# endif
#endif

#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/io/pcd_io.h>

#if defined(__clang__)
# pragma clang diagnostic pop
#endif


#include "plane_seg/PlaneFitter.hpp"
#include "plane_seg/PlaneSegmenter.hpp"
#include "plane_seg/RectangleFitter.hpp"

using namespace planeseg;
using namespace std::chrono_literals; // to recognize ms in sleep for


BlockFitter::
BlockFitter() {
  setSensorPose(Eigen::Vector3f(0,0,0), Eigen::Vector3f(1,0,0));
  setBlockDimensions(Eigen::Vector3f(15+3/8.0, 15+5/8.0, 5+5/8.0)*0.0254);
  setDownsampleResolution(0.01);
  setRemoveGround(true);
  setGroundBand(1e10,1e10);
  setHeightBand(0.05, 1.0);
  setMaxRange(3.0);
  setMaxAngleFromHorizontal(45);
  setMaxEstimationError(0.02); // RANSAC
  setMaxIterations(1000); // RANSAC
  // setMaxAngleOfPlaneSegmenter(5);
  setAreaThresholds(0.5, 1.5);
  setRectangleFitAlgorithm(RectangleFitAlgorithm::MinimumArea);
  setDebug(true);
  setVisual(false);
}

void BlockFitter::
setSensorPose(const Eigen::Vector3f& iOrigin,
              const Eigen::Vector3f& iLookDir) {
  mOrigin = iOrigin;
  mLookDir = iLookDir;
}

void BlockFitter::
setBlockDimensions(const Eigen::Vector3f& iDimensions) {
  mBlockDimensions = iDimensions;
}

void BlockFitter::
setDownsampleResolution(const float iRes) {
  mDownsampleResolution = iRes;
}

void BlockFitter::
setRemoveGround(const bool iVal) {
  mRemoveGround = iVal;
}

void BlockFitter::
setGroundBand(const float iMinZ, const float iMaxZ) {
  mMinGroundZ = iMinZ;
  mMaxGroundZ = iMaxZ;
}


void BlockFitter::
setHeightBand(const float iMinHeight, const float iMaxHeight) {
  mMinHeightAboveGround = iMinHeight;
  mMaxHeightAboveGround = iMaxHeight;
}

void BlockFitter::
setMaxRange(const float iRange) {
  mMaxRange = iRange;
}

void BlockFitter::
setMaxAngleFromHorizontal(const float iDegrees) {
  mMaxAngleFromHorizontal = iDegrees;
}

void BlockFitter::
setMaxAngleOfPlaneSegmenter(const float iDegrees) {
  mMaxAngleOfPlaneSegmenter = iDegrees;
}

void BlockFitter::
setMaxIterations(const int iIters) {
  mMaxIterations = iIters;
}

void BlockFitter::
setMaxEstimationError(const float iDist) {
  mMaxEstimationError = iDist;
}

void BlockFitter::
setAreaThresholds(const float iMin, const float iMax) {
  mAreaThreshMin = iMin;
  mAreaThreshMax = iMax;
}

void BlockFitter::
setRectangleFitAlgorithm(const RectangleFitAlgorithm iAlgo) {
  mRectangleFitAlgorithm = iAlgo;
}

void BlockFitter::
setCloud(const LabeledCloud::Ptr& iCloud) {
  mCloud = iCloud;
}

void BlockFitter::
setDebug(const bool iVal) {
  mDebug = iVal;
}

void BlockFitter::
setVisual(const bool iVal) {
  mVisual = iVal;
}

BlockFitter::Result BlockFitter::
go() {
  if (mDebug) {
    std::cout << "******* begin plane fitting *******" << std::endl;
  }
  auto global_t0 = std::chrono::high_resolution_clock::now();

  Result result;
  result.mSuccess = false;

  if (mCloud->size() < 100) return result;

  // ---------------- filtter ----------------
  // voxelize
  LabeledCloud::Ptr cloud(new LabeledCloud());
  pcl::VoxelGrid<pcl::PointXYZL> voxelGrid;
  voxelGrid.setInputCloud(mCloud);
  voxelGrid.setLeafSize(mDownsampleResolution, mDownsampleResolution,
                        mDownsampleResolution);
  voxelGrid.filter(*cloud);
  for (int i = 0; i < (int)cloud->size(); ++i) cloud->points[i].label = i;

  // // passthrough
  // pcl::PassThrough<pcl::PointXYZL> passThrough;
  // passThrough.setInputCloud (cloud);
  // // passThrough.setKeepOrganized(true);
  // passThrough.setFilterFieldName ("x");
  // passThrough.setFilterLimits (-3, 3);
  // passThrough.filter (*cloud);

  // passThrough.setInputCloud(cloud);
  // passThrough.setFilterFieldName ("y");
  // passThrough.setFilterLimits (-3, 3);
  // passThrough.filter (*cloud);
  
  // passThrough.setInputCloud(cloud);
  // passThrough.setFilterFieldName ("z");
  // passThrough.setFilterLimits (-3, 3);
  // passThrough.filter (*cloud);

  // // crop 3m
  // pcl::CropBox<pcl::PointXYZL> cropBox;
  // cropBox.setInputCloud(cloud);
  // Eigen::Vector4f max_pt;
  // Eigen::Vector4f min_pt;
  // max_pt << 3, 3, 3, 1;
  // min_pt << -3, -3, -3, 1;
  // cropBox.setMax(max_pt);
  // cropBox.setMin(min_pt);
  // // cropBox.setKeepOrganized(true); // some points are filled by NaN
  // cropBox.filter(*cloud);

  std::cout << "voxelized cloud structure: " << cloud->width << ", " << cloud->height << std::endl;

  if (mDebug) {
    std::cout << "Original cloud size " << mCloud->size() << std::endl;
    std::cout << "Voxelized cloud size " << cloud->size() << std::endl;
    pcl::io::savePCDFileBinary("cloud_full.pcd", *cloud);
  }

  if (cloud->size() < 100) return result;

  // pose
  cloud->sensor_origin_.head<3>() = mOrigin;
  cloud->sensor_origin_[3] = 1;
  Eigen::Vector3f rz = mLookDir;
  Eigen::Vector3f rx = rz.cross(Eigen::Vector3f::UnitZ());
  Eigen::Vector3f ry = rz.cross(rx);
  Eigen::Matrix3f rotation;
  rotation.col(0) = rx.normalized();
  rotation.col(1) = ry.normalized();
  rotation.col(2) = rz.normalized();
  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  pose.linear() = rotation;
  pose.translation() = mOrigin;

  // ---------------- normal estimation ----------------
  auto t0 = std::chrono::high_resolution_clock::now();
  if (mDebug) {
    std::cout << "computing normals..." << std::flush;
  }

  // normal estimation by PCL library
  NormalCloud::Ptr normals(new NormalCloud());
  pcl::NormalEstimationOMP<pcl::PointXYZL, pcl::Normal> norm_est;
  norm_est.setKSearch (25); // best planes: 10 best clustering: 25
  norm_est.setInputCloud (cloud);
  norm_est.compute (*normals);

  for (int i = 0; i < (int)normals->size(); ++i) {
    if (normals->points[i].normal_z<0) {
      normals->points[i].normal_x = -normals->points[i].normal_x;
      normals->points[i].normal_y = -normals->points[i].normal_y;
      normals->points[i].normal_z = -normals->points[i].normal_z;
    }
  }
  
  // // normal estimation by PCL library
  // pcl::NormalEstimation<pcl::PointXYZL, pcl::Normal> ne;
  // ne.setInputCloud (cloud);
  // pcl::search::KdTree<pcl::PointXYZL>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZL> ());
  // ne.setSearchMethod (tree);
  // NormalCloud::Ptr normals(new NormalCloud());
  // ne.setRadiusSearch (0.1);
  // ne.compute (*normals);

  // for (int i = 0; i < (int)normals->size(); ++i) {
  //   if (normals->points[i].normal_z<0) {
  //     normals->points[i].normal_x = -normals->points[i].normal_x;
  //     normals->points[i].normal_y = -normals->points[i].normal_y;
  //     normals->points[i].normal_z = -normals->points[i].normal_z;
  //   }
  // }

  // // visualizer
  // pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  // viewer->setBackgroundColor (0, 0, 0);
  // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZL> rgb(cloud);
  // viewer->addPointCloud<pcl::PointXYZL> (cloud, rgb, "sample cloud");
  // viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  // viewer->addPointCloudNormals<pcl::PointXYZL, pcl::Normal> (cloud, normals, 10, 0.05, "normals");
  // viewer->addCoordinateSystem (1.0);
  // viewer->initCameraParameters ();
  // while (!viewer->wasStopped ())
  // {
  //   viewer->spinOnce (100);
  //   std::this_thread::sleep_for(100ms);
  // }
  
  if (mDebug) {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0);
    std::cout << "finished in " << dt.count()/1e3 << " sec" << std::endl;
  }

  // ---------------- filt non-horizontal points ----------------
  t0 = std::chrono::high_resolution_clock::now();
  if (mDebug) {
    std::cout << "filt non-horizontal points..." << std::flush;
  }

  const float maxNormalAngle = mMaxAngleFromHorizontal*M_PI/180;
  LabeledCloud::Ptr tempCloud(new LabeledCloud());
  NormalCloud::Ptr tempNormals(new NormalCloud());
  tempCloud->reserve(normals->size());
  tempNormals->reserve(normals->size());
  for (int i = 0; i < (int)normals->size(); ++i) {
    // const auto& norm = normals->points[i];
    // Eigen::Vector3f normal(norm.normal_x, norm.normal_y, norm.normal_z);
    // float angle = std::acos(normals->points[i].normal_z);  //std::acos(normal[2]);
    float angle = std::acos(std::abs(normals->points[i].normal_z));
    if (angle > maxNormalAngle) continue;
    tempCloud->push_back(cloud->points[i]);
    tempNormals->push_back(normals->points[i]);
  }
  std::swap(tempCloud, cloud);
  std::swap(tempNormals, normals);
  result.filttedCloud = cloud;

  if (mDebug) {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0);
    std::cout << "finished in " << dt.count()/1e3 << " sec" << std::endl;

    std::cout << "Horizontal points remaining " << cloud->size() << std::endl;
    pcl::io::savePCDFileBinary("cloud.pcd", *cloud);
    pcl::io::savePCDFileBinary("robust_normals.pcd", *normals);
  }

  // ---------------- clustering by distance and curvature ----------------
  t0 = std::chrono::high_resolution_clock::now();
  if (mDebug) {
    std::cout << "clustering by distance and curvature..." << std::flush;
  }

  PlaneSegmenter segmenter;
  segmenter.setData(cloud, normals);
  segmenter.setSearchRadius(0.03);
  segmenter.setMaxAngle(mMaxAngleOfPlaneSegmenter);
  segmenter.setMinPoints(200);
  PlaneSegmenter::Result segmenterResult = segmenter.go();

  if (mDebug) {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0);
    std::cout << "finished in " << dt.count()/1e3 << " sec" << std::endl;
  }

  if (mVisual) {
    // visualize cluster results
    std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > cloud_ptrs;
    for (int j = 0; j < (int)segmenterResult.mClusterNum; ++j) {
        pcl::PointCloud<pcl::PointXYZ> clusterCloud;
        for (int i = 0; i < (int)cloud->size(); ++i){
          if (segmenterResult.mLabels[i] == j) {
            pcl::PointXYZ pt;
            pt.x = cloud->points[i].x;
            pt.y = cloud->points[i].y;
            pt.z = cloud->points[i].z;
            clusterCloud.points.push_back(pt);
          }
        }
        clusterCloud.height = clusterCloud.points.size();
        clusterCloud.width = 1;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr;
        cloud_ptr = clusterCloud.makeShared();
        cloud_ptrs.push_back(cloud_ptr);
    }
    
    std::vector<double> colors_ = {
        51/255.0, 160/255.0, 44/255.0,  //0
        166/255.0, 206/255.0, 227/255.0,
        178/255.0, 223/255.0, 138/255.0,//6
        31/255.0, 120/255.0, 180/255.0,
        251/255.0, 154/255.0, 153/255.0,// 12
        227/255.0, 26/255.0, 28/255.0,
        253/255.0, 191/255.0, 111/255.0,// 18
        106/255.0, 61/255.0, 154/255.0,
        255/255.0, 127/255.0, 0/255.0, // 24
        202/255.0, 178/255.0, 214/255.0,
        1.0, 0.0, 0.0, // red // 30
        0.0, 1.0, 0.0, // green
        0.0, 0.0, 1.0, // blue// 36
        1.0, 1.0, 0.0,
        1.0, 0.0, 1.0, // 42
        0.0, 1.0, 1.0,
        0.5, 1.0, 0.0,
        1.0, 0.5, 0.0,
        0.5, 0.0, 1.0,
        1.0, 0.0, 0.5,
        0.0, 0.5, 1.0,
        0.0, 1.0, 0.5,
        1.0, 0.5, 0.5,
        0.5, 1.0, 0.5,
        0.5, 0.5, 1.0,
        0.5, 0.5, 1.0,
        0.5, 1.0, 0.5,
        0.5, 0.5, 1.0};

    pcl::PointCloud<pcl::PointXYZRGB> combined_cloud;
    for (int i = 0; i < (int)cloud_ptrs.size(); ++i) {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::copyPointCloud(*cloud_ptrs[i], *cloud_rgb);

      int nColor = i % (colors_.size()/3);
      double r = colors_[nColor*3]*255.0;
      double g = colors_[nColor*3+1]*255.0;
      double b = colors_[nColor*3+2]*255.0;
      for (int j = 0; j < (int)cloud_rgb->points.size (); j++){
          cloud_rgb->points[j].r = r;
          cloud_rgb->points[j].g = g;
          cloud_rgb->points[j].b = b;
      }
      combined_cloud += *cloud_rgb;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_cloud_ptr;
    combined_cloud_ptr = combined_cloud.makeShared();
    pcl::visualization::CloudViewer viewer2 ("Cloud Viewer 2");
    viewer2.showCloud(combined_cloud_ptr);  
    while (!viewer2.wasStopped ()) {}

  }

  // ---------------- fit a plane for each cluster ----------------
  t0 = std::chrono::high_resolution_clock::now();
  if (mDebug) {
    std::cout << "fit each cluster to a plane and then to a rectangle..." << std::flush;
  }

  std::vector<RectangleFitter::Result> results;
  results.reserve(segmenterResult.mClusterNum);
  for (int i = 0; i < (int)segmenterResult.mClusterNum; ++i) {
    // std::cout << "\ncluster " << i << " with size of " << segmenterResult.mClusters[i].size() << std::endl;
    PlaneFitter planeFitter;
    planeFitter.setMaxIterations(mMaxIterations);
    planeFitter.setMaxDistance(mMaxEstimationError);
    planeFitter.setRefineUsingInliers(true);
    PlaneFitter::Result planeFitterRes = planeFitter.go(segmenterResult.mClusters[i]);
    // std::cout << "\nFitted plane: " << planeFitterRes.mPlane << std::endl;

    RectangleFitter fitter;
    fitter.setDimensions(mBlockDimensions.head<2>());
    fitter.setAlgorithm((RectangleFitter::Algorithm)mRectangleFitAlgorithm);
    fitter.setData(segmenterResult.mClusters[i], planeFitterRes.mPlane);
    auto result = fitter.go();
    results.push_back(result);

  }

  if (mDebug) {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0);
    std::cout << "finished in " << dt.count()/1e3 << " sec" << std::endl;
  }

  // ---------------- get blocks ----------------
  t0 = std::chrono::high_resolution_clock::now();
  if (mDebug) {
    std::cout << "getting blocks..." << std::flush;
  }

  for (int i = 0; i < (int)results.size(); ++i) {
    const auto& res = results[i];

    // These seems to filter out hulls which are not of similar size to the DRC
    // blocks. I think this irrelevent for general use
    // if (mBlockDimensions.head<2>().norm() > 1e-5) {
    //   float areaRatio = mBlockDimensions.head<2>().prod()/res.mConvexArea;
    //   // std::cout << mBlockDimensions.transpose() << " | " << res.mConvexArea << " | " << areaRatio << " " << i << "\n";
    //   if ((areaRatio < mAreaThreshMin) ||
    //       (areaRatio > mAreaThreshMax)) continue;
    // }

    Block block;
    block.mSize << res.mSize[0], res.mSize[1], mBlockDimensions[2];
    block.mPose = res.mPose;
    block.mPose.translation() -=
      block.mPose.rotation().col(2)*mBlockDimensions[2]/2;
    block.mHull = res.mConvexHull;
    result.mBlocks.push_back(block);
  }

  if (mDebug) {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0);
    std::cout << "finished in " << dt.count()/1e3 << " sec" << std::endl;

    std::cout << "Surviving blocks: " << result.mBlocks.size() << std::endl;
  }

  result.mSuccess = true;

  if (mDebug) {
    auto global_t1 = std::chrono::high_resolution_clock::now();
    auto global_dt = std::chrono::duration_cast<std::chrono::milliseconds>(global_t1-global_t0);
    std::cout << "******* plane segmentation finished in " << global_dt.count()/1e3 << " sec *******" << std::endl;
  }

  return result;
}
