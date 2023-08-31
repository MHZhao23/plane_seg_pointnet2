#include "plane_seg/Preprocessing.hpp"

#include <unistd.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <ros/package.h>

#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

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
#include <sensor_msgs/PointCloud2.h>

#if defined(__clang__)
# pragma clang diagnostic pop
#endif

using namespace planeseg;
using namespace std::chrono_literals; // to recognize ms in sleep for


Preprocessing::
Preprocessing() {
    setSensorPose(Eigen::Vector3f(0,0,0), Eigen::Vector3f(1,0,0));
    setDownsampleResolution(0.01);
    setMaxAngleFromHorizontal(60);
    setDebug(true);
    setVisual(false);
}

void Preprocessing::
setSensorPose(const Eigen::Vector3f& iOrigin,
                const Eigen::Vector3f& iLookDir) {
    mOrigin = iOrigin;
    mLookDir = iLookDir;
}

void Preprocessing::
setDownsampleResolution(const float iRes) {
    mDownsampleResolution = iRes;
}

void Preprocessing::
setMaxAngleFromHorizontal(const float iDegrees) {
    mMaxAngleFromHorizontal = iDegrees;
}

void Preprocessing::
setCloud(const LabeledCloud::Ptr& iCloud) {
    mCloud = iCloud;
}

void Preprocessing::
setFrame(const std::string& iCloudFrame) {
    mCloudFrame = iCloudFrame;
}

void Preprocessing::
setDebug(const bool iVal) {
    mDebug = iVal;
}

void Preprocessing::
setVisual(const bool iVal) {
    mVisual = iVal;
}

LabeledCloud::Ptr Preprocessing::
go() {
    std::cout << "\n\n******* begin pre-processing *******" << std::endl;
    auto global_t0 = std::chrono::high_resolution_clock::now();

    if (mCloud->size() < 100) return mCloud;

    // ---------------- filtter ----------------
    // voxelize
    LabeledCloud::Ptr cloud(new LabeledCloud());
    pcl::VoxelGrid<pcl::PointXYZL> voxelGrid;
    voxelGrid.setInputCloud(mCloud);
    voxelGrid.setLeafSize(mDownsampleResolution, mDownsampleResolution,
                          mDownsampleResolution);
    voxelGrid.filter(*cloud);
    for (int i = 0; i < (int)cloud->size(); ++i) cloud->points[i].label = i;

    // crop 2m
    int crop_xdim = 2;
    int crop_ydim = 2;
    int crop_zdim = 2;
    pcl::CropBox<pcl::PointXYZL> cropBox;
    cropBox.setInputCloud(cloud);
    Eigen::Vector4f max_pt;
    Eigen::Vector4f min_pt;
    max_pt << crop_xdim, crop_ydim, crop_zdim, 1;
    min_pt << -crop_xdim, -crop_ydim, -crop_zdim, 1;
    cropBox.setMax(max_pt);
    cropBox.setMin(min_pt);
    cropBox.filter(*cloud);

    if (mDebug) {
        std::cout << "Original cloud size " << mCloud->size() << std::endl;
        std::cout << "Voxelized cloud size " << cloud->size() << std::endl;
    }

    if (mCloud->size() < 100) return mCloud;

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

    if (mDebug) {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0);
        std::cout << "finished in " << dt.count()/1e3 << " sec" << std::endl;
        std::cout << "Horizontal points remaining " << cloud->size() << std::endl;
    }

    auto global_t1 = std::chrono::high_resolution_clock::now();
    auto global_dt = std::chrono::duration_cast<std::chrono::milliseconds>(global_t1 - global_t0);
    std::cout << "finished in " << global_dt.count()/1e3 << " sec" << std::endl;

    return cloud;

}
