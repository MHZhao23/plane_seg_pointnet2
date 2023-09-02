#include "plane_seg/PlaneSegmenter.hpp"

#include <queue>
#include <pcl/search/kdtree.h>

#include "plane_seg/IncrementalPlaneEstimator.hpp"

using namespace planeseg;

PlaneSegmenter::
PlaneSegmenter() {
  // setMaxAngle(30);
  // setSearchRadius(0.03);
  // setMinPoints(500);
}

void PlaneSegmenter::
setData(const LabeledCloud::Ptr& iCloud,
        const NormalCloud::Ptr& iNormals) {
  mCloud = iCloud;
  mNormals = iNormals;
}

void PlaneSegmenter::
setMaxError(const float iError) {
  mMaxError = iError;
}

void PlaneSegmenter::
setMaxAngle(const float iAngle) {
  mMaxAngle = iAngle*M_PI/180;
}

void PlaneSegmenter::
setSearchRadius(const float iRadius) {
  mSearchRadius = iRadius;
}


void PlaneSegmenter::
setMinPoints(const int iMin) {
  mMinPoints = iMin;
}

PlaneSegmenter::Result PlaneSegmenter::
go() {
  Result result;
  const int n = mCloud->size();

  // create kdtree and get nearest neighbors list
  pcl::search::KdTree<Point>::Ptr tree (new pcl::search::KdTree<Point>());
  tree->setInputCloud(mCloud);
  std::vector<std::vector<int>> neighbors(n);
  std::vector<float> distances;
  for (int i = 0; i < n; ++i) { 
    tree->radiusSearch(i, mSearchRadius, neighbors[i], distances);
    auto& neigh = neighbors[i];
    std::vector<std::pair<int,float>> pairs(neigh.size());
    for (int j = 0; j < (int)neigh.size(); ++j) {
      pairs[j].first = neigh[j];
      pairs[j].second = distances[j];
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<int,float>& iA,
                 const std::pair<int,float>& iB){
                return iA.second<iB.second;});
    for (int j = 0; j < (int)neigh.size(); ++j) {
      neigh[j] = pairs[j].first;
    }
  }

  // create labels
  std::vector<int> labels(n); // which cluster the point belongs to (0, 1, ... curLabel)
  std::fill(labels.begin(), labels.end(), -1);
  std::vector<int> pointsLabels(n); // which large component the point belongs to (-1, 0, 1, ... vaildLabel)
  std::fill(pointsLabels.begin(), pointsLabels.end(), -1);
  pointsClustering pointsCluster;
  int curLabel = 0;
  int vaildLabel = 0;
  // std::vector<int> clusterLables; // 0, 1, 2, 3, 4, ...
  std::vector<int> validLables; // 0, -1, 1, -1, 2 ...
  std::deque<int> workQueue;

  // int ppoint_num = 0;
  // for (int i = 0; i < n; ++i) {
  //   // std::cout << mCloud->points[i].x << ", " << mCloud->points[i].y << ", " << mCloud->points[i].z << ", " << mCloud->points[i].label << "/// " << std::flush;
  //   if (mCloud->points[i].label > 0) ppoint_num += 1;
  // }
  // std::cout<< "number of plane points: " << ppoint_num << std::endl;
  auto processPoint = [&](const int iIndex) {
      if (labels[iIndex] < 0) {
      // std::cout << iIndex << ", " << std::flush;
      const auto& cloudNorm = mNormals->points[iIndex];
      const Eigen::Vector3f norm(cloudNorm.normal_x, cloudNorm.normal_y,
                                cloudNorm.normal_z);
      if (pointsCluster.tryPoint(norm, mMaxAngle)) {
        labels[iIndex] = curLabel;
        pointsCluster.addPoint(norm);
        for (const auto idx : neighbors[iIndex]) {
          if (labels[idx] < 0) workQueue.push_back(idx);
        }
        return true;
      }
      return false;
    }
    return false;
  };

  // iterate over points
  for (int idx = 0; idx < n; ++idx) {
    if (labels[idx] >= 0) continue;

    // start new component
    pointsCluster.reset();
    workQueue.clear();
    workQueue.push_back(idx);

    while (workQueue.size() > 0) {
      processPoint(workQueue.front());
      workQueue.pop_front();
    }

    // add new cluster
    int clusterSize = pointsCluster.getNumPoints();
    if (clusterSize > mMinPoints) {
      // std::cout << "get cluster " << curLabel << " with " << clusterSize << " points " << std::endl;
      // clusterLables.push_back(curLabel);
      validLables.push_back(vaildLabel);
      // result.mClusterSizes.push_back(clusterSize);
      vaildLabel += 1;
    }
    else {
      // clusterLables.push_back(-1);
      validLables.push_back(-1);
    }
    ++curLabel;
  }

  // for (int i = 0; i < n; ++i) {
  //   pointsLabels[i] = validLables[labels[i]];
  // }

  // std::cout << "\nclusterLables: " << std::endl;
  // for (int i = 0; i < curLabel; ++i) std::cout << clusterLables[i] << ", " << std::flush;

  // std::cout << "\nvalidLables: " << std::endl;
  // for (int i = 0; i < curLabel; ++i) std::cout << validLables[i] << ", " << std::flush;

  // std::cout << "\nlabels: " << std::endl;
  // for (int i = 0; i < n; ++i) std::cout << labels[i] << ", " << std::flush;

  // std::cout << "\npointLabels: " << std::endl;
  // for (int i = 0; i < n; ++i) std::cout << pointsLabels[i] << ", " << std::flush;

  std::unordered_map<int,std::vector<Eigen::Vector3f>> cloudMap;
  for (int i = 0; i < n; ++i) {
    int iLabel = validLables[labels[i]];
    pointsLabels[i] = iLabel;
    if (iLabel >= 0) cloudMap[iLabel].push_back(mCloud->points[i].getVector3fMap());
  }

  std::vector<MatrixX3f> clusterPoints;
  int clusterLabel = 0;
  clusterPoints.reserve(cloudMap.size());
  for (auto it : cloudMap) {
    int singleCloudSize = it.second.size();
    // std::cout << "\tcluster size: " << singleCloudSize << std::flush;
    MatrixX3f singleCloud;
    singleCloud.resize(singleCloudSize,3);
    for (int i = 0; i < singleCloudSize; ++i) singleCloud.row(i) = it.second[i];
    result.mClusterSizes.push_back(singleCloudSize);
    result.mClusterLabels.push_back(clusterLabel);
    clusterPoints.push_back(singleCloud);
    clusterLabel ++;
  }

  result.mClusters = clusterPoints;
  result.mLabels = pointsLabels;
  result.mClusterNum = vaildLabel;

  return result;
}
