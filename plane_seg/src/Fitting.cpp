#include "plane_seg/Fitting.hpp"

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


Fitting::
Fitting() {
  setBlockDimensions(Eigen::Vector3f(15+3/8.0, 15+5/8.0, 5+5/8.0)*0.0254);
  setMaxEstimationError(0.02); // RANSAC
  setMaxIterations(1000); // RANSAC
  setRectangleFitAlgorithm(RectangleFitAlgorithm::MinimumArea);
  setDebug(true);
  setVisual(false);
}

void Fitting::
setBlockDimensions(const Eigen::Vector3f& iDimensions) {
  mBlockDimensions = iDimensions;
}

void Fitting::
setMaxAngleOfPlaneSegmenter(const float iDegrees) {
  mMaxAngleOfPlaneSegmenter = iDegrees;
}

void Fitting::
setMaxIterations(const int iIters) {
  mMaxIterations = iIters;
}

void Fitting::
setMaxEstimationError(const float iDist) {
  mMaxEstimationError = iDist;
}

void Fitting::
setRectangleFitAlgorithm(const RectangleFitAlgorithm iAlgo) {
  mRectangleFitAlgorithm = iAlgo;
}

void Fitting::
setCloud(const LabeledCloud::Ptr& iCloud) {
  mCloud = iCloud;
}

void Fitting::
setFrame(const std::string& iCloudFrame) {
    mCloudFrame = iCloudFrame;
}

void Fitting::
setDebug(const bool iVal) {
  mDebug = iVal;
}

void Fitting::
setVisual(const bool iVal) {
  mVisual = iVal;
}

Fitting::Result Fitting::
go() {
  std::cout << "\n\n******* begin plane segmentation *******" << std::endl;
  auto global_t0 = std::chrono::high_resolution_clock::now();

  Result result;
  result.mSuccess = false;

  // ---------------- normal estimation ----------------
  auto t0 = std::chrono::high_resolution_clock::now();
  if (mDebug) {
      std::cout << "computing normals..." << std::flush;
  }

  // normal estimation by PCL library
  NormalCloud::Ptr normals(new NormalCloud());
  pcl::NormalEstimationOMP<pcl::PointXYZL, pcl::Normal> norm_est;
  norm_est.setKSearch (25); // best planes: 10 best clustering: 25
  norm_est.setInputCloud (mCloud);
  norm_est.compute (*normals);

  for (int i = 0; i < (int)normals->size(); ++i) {
      if (normals->points[i].normal_z<0) {
          normals->points[i].normal_x = -normals->points[i].normal_x;
          normals->points[i].normal_y = -normals->points[i].normal_y;
          normals->points[i].normal_z = -normals->points[i].normal_z;
      }
  }

  if (mDebug) {
      auto t1 = std::chrono::high_resolution_clock::now();
      auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0);
      std::cout << "finished in " << dt.count()/1e3 << " sec" << std::endl;
  }

  // ---------------- clustering by distance and curvature ----------------
  if (mDebug) {
    std::cout << "clustering by distance and curvature..." << std::flush;
  }

  PlaneSegmenter segmenter;
  segmenter.setData(mCloud, normals);
  segmenter.setSearchRadius(0.03);
  segmenter.setMaxAngle(mMaxAngleOfPlaneSegmenter);
  segmenter.setMinPoints(150);
  PlaneSegmenter::Result segmenterResult = segmenter.go();

  // // visualize cluster results
  // std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > cloud_ptrs;
  // for (int j = 0; j < (int)segmenterResult.mClusterNum; ++j) {
  //     pcl::PointCloud<pcl::PointXYZ> clusterCloud;
  //     for (int i = 0; i < (int)mCloud->size(); ++i){
  //       if (segmenterResult.mLabels[i] == j) {
  //         pcl::PointXYZ pt;
  //         pt.x = mCloud->points[i].x;
  //         pt.y = mCloud->points[i].y;
  //         pt.z = mCloud->points[i].z;
  //         clusterCloud.points.push_back(pt);
  //       }
  //     }
  //     clusterCloud.height = clusterCloud.points.size();
  //     clusterCloud.width = 1;

  //     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr;
  //     cloud_ptr = clusterCloud.makeShared();
  //     cloud_ptrs.push_back(cloud_ptr);
  // }
  
  // std::vector<double> colors_ = {
  //     51/255.0, 160/255.0, 44/255.0,  //0
  //     166/255.0, 206/255.0, 227/255.0,
  //     178/255.0, 223/255.0, 138/255.0,//6
  //     31/255.0, 120/255.0, 180/255.0,
  //     251/255.0, 154/255.0, 153/255.0,// 12
  //     227/255.0, 26/255.0, 28/255.0,
  //     253/255.0, 191/255.0, 111/255.0,// 18
  //     106/255.0, 61/255.0, 154/255.0,
  //     255/255.0, 127/255.0, 0/255.0, // 24
  //     202/255.0, 178/255.0, 214/255.0,
  //     1.0, 0.0, 0.0, // red // 30
  //     0.0, 1.0, 0.0, // green
  //     0.0, 0.0, 1.0, // blue// 36
  //     1.0, 1.0, 0.0,
  //     1.0, 0.0, 1.0, // 42
  //     0.0, 1.0, 1.0,
  //     0.5, 1.0, 0.0,
  //     1.0, 0.5, 0.0,
  //     0.5, 0.0, 1.0,
  //     1.0, 0.0, 0.5,
  //     0.0, 0.5, 1.0,
  //     0.0, 1.0, 0.5,
  //     1.0, 0.5, 0.5,
  //     0.5, 1.0, 0.5,
  //     0.5, 0.5, 1.0,
  //     0.5, 0.5, 1.0,
  //     0.5, 1.0, 0.5,
  //     0.5, 0.5, 1.0};

  // pcl::PointCloud<pcl::PointXYZRGB> combined_cloud;
  // for (int i = 0; i < (int)cloud_ptrs.size(); ++i) {
  //   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
  //   pcl::copyPointCloud(*cloud_ptrs[i], *cloud_rgb);

  //   int nColor = i % (colors_.size()/3);
  //   double r = colors_[nColor*3]*255.0;
  //   double g = colors_[nColor*3+1]*255.0;
  //   double b = colors_[nColor*3+2]*255.0;
  //   for (int j = 0; j < (int)cloud_rgb->points.size (); j++){
  //       cloud_rgb->points[j].r = r;
  //       cloud_rgb->points[j].g = g;
  //       cloud_rgb->points[j].b = b;
  //   }
  //   combined_cloud += *cloud_rgb;
  // }

  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_cloud_ptr;
  // combined_cloud_ptr = combined_cloud.makeShared();
  // pcl::visualization::CloudViewer viewer2 ("Cloud Viewer 2");
  // viewer2.showCloud(combined_cloud_ptr);  
  // while (!viewer2.wasStopped ()) {}

  if (mDebug) {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0);
    std::cout << "finished in " << dt.count()/1e3 << " sec" << std::endl;
  }

  // ---------------- fit a plane for each cluster ----------------
  t0 = std::chrono::high_resolution_clock::now();
  if (mDebug) {
    std::cout << "fit each cluster to a plane and then to a rectangle..." << std::flush;
  }

  int numClusters = segmenterResult.mClusterNum;
  Eigen::MatrixXf normDiffs(numClusters, numClusters);
  // Eigen::MatrixXf offsetDiffs(numClusters, numClusters);
  std::vector<Eigen::Vector4f> planeResults;
  std::cout << "numClusters: " << numClusters << std::endl;
  for (int i = 0; i < numClusters; ++i) {
    // std::cout << "\ncluster " << i << " with size of " << segmenterResult.mClusters[i].size() << std::endl;
    PlaneFitter planeFitter;
    planeFitter.setMaxIterations(mMaxIterations);
    planeFitter.setMaxDistance(mMaxEstimationError);
    planeFitter.setRefineUsingInliers(true);
    PlaneFitter::Result planeFitterRes = planeFitter.go(segmenterResult.mClusters[i]);
    planeResults.push_back(planeFitterRes.mPlane);
    // std::cout << "\nFitted plane: " << planeFitterRes.mPlane << std::endl;

    for (int j = 0; j < i; ++j) {
      Eigen::Vector3f iNorm = planeFitterRes.mPlane.head<3>();
      Eigen::Vector3f jNorm = planeResults[j].head<3>();
      float ijNormDiff = std::acos(std::abs(iNorm[0] * jNorm[0] + iNorm[1] * jNorm[1] + iNorm[2] * jNorm[2]));
      // std::cout << i << ", " << j << ", " << ijNormDiff << "; " << std::flush;
      normDiffs(i, j) = ijNormDiff;
      normDiffs(j, i) = ijNormDiff;

      // float iOffset = planeFitterRes.mPlane[3];
      // float jOffset = planeResults[j][3];
      // float ijOffsetDiff = std::abs(iOffset - jOffset);
      // // std::cout << i << ", " << j << ", " << ijOffsetDiff << "; " << std::endl;
      // offsetDiffs(i, j) = ijOffsetDiff;
      // offsetDiffs(j, i) = ijOffsetDiff;
    }
  }

  float maxNormDiff = 3*M_PI/180;
  // float maxOffsetDiff = 0.05;
  std::vector<int> mergedClusterIdx;
  for (int i = 0; i < numClusters; ++i) {
    for (int j = (i+1); j < numClusters; ++j) {
      if (normDiffs(i, j) < maxNormDiff) {
        // std::cout << i << ", " << j << "; " << std::endl;
        // merge cluster i and j
        int iClusterSize = segmenterResult.mClusterSizes[i];
        int jClusterSize = segmenterResult.mClusterSizes[j];
        int mergedSize = iClusterSize + jClusterSize;
        Eigen::MatrixX3f mergedClusters(mergedSize, 3);
        for (int k = 0; k < iClusterSize; k++) mergedClusters.row(k) = segmenterResult.mClusters[i].row(k);
        for (int k = iClusterSize; k < mergedSize; k++) mergedClusters.row(k) = segmenterResult.mClusters[j].row(k-iClusterSize);
        // update segmenterResult
        segmenterResult.mClusters.erase (segmenterResult.mClusters.begin() + i);
        segmenterResult.mClusters.insert(segmenterResult.mClusters.begin() + i, mergedClusters);
        // segmenterResult.mClusters[i] = mergedClusters;
        segmenterResult.mClusterSizes[i] = mergedSize;
        mergedClusterIdx.push_back(j);
      }
    }
  }

  int erasedNum = 0;
  if (mergedClusterIdx.size() != 0) {
    for (auto & idx : mergedClusterIdx) {
      segmenterResult.mClusters.erase (segmenterResult.mClusters.begin() + (idx-erasedNum));
      segmenterResult.mLabels.erase (segmenterResult.mLabels.begin() + (idx-erasedNum));
      segmenterResult.mClusterSizes.erase (segmenterResult.mClusterSizes.begin() + (idx-erasedNum));
      erasedNum++; 
    }
    segmenterResult.mClusterNum = segmenterResult.mClusters.size();
  }

  numClusters = segmenterResult.mClusterNum;
  std::vector<Eigen::Vector4f> mergedPlaneResults;
  for (int i = 0; i < numClusters; ++i) {
    // std::cout << "\ncluster " << i << " with size of " << segmenterResult.mClusters[i].size() << std::endl;
    PlaneFitter planeFitter;
    planeFitter.setMaxIterations(mMaxIterations);
    planeFitter.setMaxDistance(mMaxEstimationError);
    planeFitter.setRefineUsingInliers(true);
    PlaneFitter::Result planeFitterRes = planeFitter.go(segmenterResult.mClusters[i]);
    mergedPlaneResults.push_back(planeFitterRes.mPlane);
  }

  std::cout << "numClusters after merging: " << segmenterResult.mClusterNum << std::endl;

  // ---------------- merge planes and rectangularize ----------------
  std::vector<RectangleFitter::Result> results;
  results.reserve(segmenterResult.mClusterNum);
  for (int i = 0; i < segmenterResult.mClusterNum; ++i) {
    RectangleFitter fitter;
    fitter.setDimensions(mBlockDimensions.head<2>());
    fitter.setAlgorithm((RectangleFitter::Algorithm)mRectangleFitAlgorithm);
    fitter.setData(segmenterResult.mClusters[i], mergedPlaneResults[i]);
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

  auto global_t1 = std::chrono::high_resolution_clock::now();
  auto global_dt = std::chrono::duration_cast<std::chrono::milliseconds>(global_t1 - global_t0);
  std::cout << "finished in " << global_dt.count()/1e3 << " sec" << std::endl;

  return result;
}
