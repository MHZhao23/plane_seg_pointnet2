#include <unistd.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <ros/package.h>
#include <std_msgs/Time.h>
#include "ros/time.h"
#include "time.h"  

#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>

// GridMapRosConverter includes cv_bridge which includes OpenCV4 which uses _Atomic
// We want to ignore this warning entirely.
#if defined(__clang__)
# pragma clang diagnostic push
#endif

#if defined(__clang__) && defined(__has_warning)
# if __has_warning( "-Wc11-extensions" )
#  pragma clang diagnostic ignored "-Wc11-extensions"
# endif
#endif

#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_ros/GridMapRosConverter.hpp>

#if defined(__clang__)
# pragma clang diagnostic pop
#endif

// tf
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>

// #include "plane_seg/Timer.hpp"
#include "plane_seg_ros/plane_seg_ros.hpp"
#include "plane_seg/BlockFitter.hpp"
#include "plane_seg/Preprocessing.hpp"
#include "plane_seg/Fitting.hpp"

// #define WITH_TIMING

#ifdef WITH_TIMING
#include <chrono>
#endif


// convenience methods
auto vecToStr = [](const Eigen::Vector3f& iVec) {
  std::ostringstream oss;
  oss << iVec[0] << ", " << iVec[1] << ", " << iVec[2];
  return oss.str();
};
auto rotToStr = [](const Eigen::Matrix3f& iRot) {
  std::ostringstream oss;
  Eigen::Quaternionf q(iRot);
  oss << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z();
  return oss.str();
};


class Pass{
  public:
    Pass(ros::NodeHandle node_);
    ~Pass() = default;

    void timeCallback(const std_msgs::Time::ConstPtr &msg);
    void planePointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);

    void preProcessCloud(const std::string& cloudFrame, planeseg::LabeledCloud::Ptr& inCloud, Eigen::Vector3f origin, Eigen::Vector3f lookDir);
    void fittingPlane(const std::string& cloudFrame, planeseg::LabeledCloud::Ptr& inCloud, Eigen::Vector3f origin, Eigen::Vector3f lookDir);

    void publishHullsAsCloud(const std::string& cloud_frame, std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_ptrs, int secs, int nsecs);
    void publishHullsAsMarkers(const std::string& cloud_frame, std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > cloud_ptrs, int secs, int nsecs);
    void publishHullsAsMarkerArray(const std::string& cloud_frame, std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > cloud_ptrs, int secs, int nsecs);
    void publishResult(const std::string& cloud_frame);

  private:
    ros::Time begin;
    ros::NodeHandle node_;
    std::vector<double> colors_;

    ros::Subscriber time_sub_, plane_cloud_sub_;
    ros::Publisher preprocessed_cloud_pub, hull_cloud_pub_, hull_markers_pub_, look_pose_pub_, hull_marker_array_pub_;

    std::string fixed_frame_ = "odom";  // Frame in which all results are published. "odom" for backwards-compatibility. Likely should be "map".

    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformListener tfListener_;

    planeseg::Fitting::Result fresult_;
};

Pass::Pass(ros::NodeHandle node_):
    node_(node_),
    tfBuffer_(ros::Duration(5.0)),
    tfListener_(tfBuffer_) {

  // // subscribe the labelled point cloud
  time_sub_ = node_.subscribe("/plane_seg_n1/start_time", 10,
                                        &Pass::timeCallback, this);
  plane_cloud_sub_ = node_.subscribe("/plane_seg_n2/plane_cloud", 10,
                                        &Pass::planePointCloudCallback, this);
                                        
  // publishing the results
  hull_cloud_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/plane_seg_n3/hull_cloud", 10);
  hull_markers_pub_ = node_.advertise<visualization_msgs::Marker>("/plane_seg_n3/hull_markers", 10);
  hull_marker_array_pub_ = node_.advertise<visualization_msgs::MarkerArray>("/plane_seg_n3/hull_marker_array", 10);
  look_pose_pub_ = node_.advertise<geometry_msgs::PoseStamped>("/plane_seg_n3/look_pose", 10);

  colors_ = {
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

}

void quat_to_euler(const Eigen::Quaterniond& q, double& roll, double& pitch, double& yaw) {
  const double q0 = q.w();
  const double q1 = q.x();
  const double q2 = q.y();
  const double q3 = q.z();
  roll = atan2(2.0*(q0*q1+q2*q3), 1.0-2.0*(q1*q1+q2*q2));
  pitch = asin(2.0*(q0*q2-q3*q1));
  yaw = atan2(2.0*(q0*q3+q1*q2), 1.0-2.0*(q2*q2+q3*q3));
}

Eigen::Vector3f convertRobotPoseToSensorLookDir(Eigen::Isometry3d robot_pose){

  Eigen::Quaterniond quat = Eigen::Quaterniond( robot_pose.rotation() );
  double r,p,y;
  quat_to_euler(quat, r, p, y);
  //std::cout << r*180/M_PI << ", " << p*180/M_PI << ", " << y*180/M_PI << " rpy in Degrees\n";

  double yaw = y;
  double pitch = -p;
  double xDir = cos(yaw)*cos(pitch);
  double yDir = sin(yaw)*cos(pitch);
  double zDir = sin(pitch);
  return Eigen::Vector3f(xDir, yDir, zDir);
}


void Pass::timeCallback(const std_msgs::Time::ConstPtr &msg) {
  begin = msg->data;
}


void Pass::planePointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg){
  planeseg::LabeledCloud::Ptr inCloud(new planeseg::LabeledCloud());
  pcl::fromROSMsg(*msg,*inCloud);

  // Look up transform from fixed frame to point cloud frame
  geometry_msgs::TransformStamped fixed_frame_to_cloud_frame_tf;
  Eigen::Isometry3d map_T_pointcloud;
  if (tfBuffer_.canTransform(fixed_frame_, msg->header.frame_id, msg->header.stamp, ros::Duration(0.0)))
  {
    fixed_frame_to_cloud_frame_tf = tfBuffer_.lookupTransform(fixed_frame_, msg->header.frame_id, msg->header.stamp, ros::Duration(0.0));
    map_T_pointcloud = tf2::transformToEigen(fixed_frame_to_cloud_frame_tf);
  }
  else
  {
    ROS_WARN_STREAM("Cannot look up transform from '" << msg->header.frame_id << "' to fixed frame ('" << fixed_frame_ <<"')");
  }

  Eigen::Vector3f origin, lookDir;
  origin << map_T_pointcloud.translation().cast<float>();
  lookDir = convertRobotPoseToSensorLookDir(map_T_pointcloud);

  fittingPlane(msg->header.frame_id, inCloud, origin, lookDir);
}


void Pass::fittingPlane(const std::string& cloudFrame, planeseg::LabeledCloud::Ptr& inCloud, Eigen::Vector3f origin, Eigen::Vector3f lookDir){
  planeseg::Fitting cfitter;
  cfitter.setFrame(cloudFrame);
  cfitter.setCloud(inCloud);
  cfitter.setDebug(false);
  cfitter.setVisual(false);
  fresult_ = cfitter.go();

  ros::Time end = ros::Time::now();
  ros::Duration dt = end - begin;
  double secs = dt.toSec();
  std::cout << "[PlaneSegmentation] took " << secs << " sec" << std::endl;
  // extern std::chrono::time_point<std::chrono::high_resolution_clock> startT;
  // auto startTime = startT;
  // auto endTime = std::chrono::high_resolution_clock::now();
  // auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
  // std::cout << std::to_string(std::time(0)) << std::endl;
  // std::cout << "[PlaneSegmentation] took " << dt.count()/1e3 << " sec" << std::endl;

  if (look_pose_pub_.getNumSubscribers() > 0) {
    Eigen::Vector3f rz = lookDir;
    Eigen::Vector3f rx = rz.cross(Eigen::Vector3f::UnitZ());
    Eigen::Vector3f ry = rz.cross(rx);
    Eigen::Matrix3f rotation;
    rotation.col(0) = rx.normalized();
    rotation.col(1) = ry.normalized();
    rotation.col(2) = rz.normalized();
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.linear() = rotation;
    pose.translation() = origin;
    Eigen::Isometry3d pose_d = pose.cast<double>();

    geometry_msgs::PoseStamped msg;
    msg.header.stamp = ros::Time(0, 0);
    msg.header.frame_id = cloudFrame;
    tf::poseEigenToMsg(pose_d, msg.pose);
    look_pose_pub_.publish(msg);
  }

  publishResult(cloudFrame);
}


void Pass::publishResult(const std::string& cloud_frame){
  // convert result to a vector of point clouds
  std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > cloud_ptrs;
  for (size_t i=0; i<(int)fresult_.mBlocks.size(); ++i){
    pcl::PointCloud<pcl::PointXYZ> cloud;
    const auto& block = fresult_.mBlocks[i];
    cloud.points.reserve(block.mHull.size());

    if (block.mHull.size() == 0) continue;
    for (size_t j =0; j < block.mHull.size(); ++j){
      pcl::PointXYZ pt;
      pt.x =block.mHull[j](0);
      pt.y =block.mHull[j](1);
      pt.z =block.mHull[j](2);
      cloud.points.push_back(pt);
    }
    cloud.height = cloud.points.size();
    cloud.width = 1;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr;
    cloud_ptr = cloud.makeShared();
    cloud_ptrs.push_back(cloud_ptr);
  }

  if (hull_cloud_pub_.getNumSubscribers() > 0) publishHullsAsCloud(cloud_frame, cloud_ptrs, 0, 0);
  if (hull_markers_pub_.getNumSubscribers() > 0) publishHullsAsMarkers(cloud_frame, cloud_ptrs, 0, 0);
  if (hull_marker_array_pub_.getNumSubscribers() > 0) publishHullsAsMarkerArray(cloud_frame, cloud_ptrs, 0, 0);
}


// combine the individual clouds into one, with a different each
void Pass::publishHullsAsCloud(const std::string& cloud_frame,
                               std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > cloud_ptrs,
                               int secs, int nsecs){
  pcl::PointCloud<pcl::PointXYZRGB> combined_cloud;
  for (size_t i=0; i<(int)cloud_ptrs.size(); ++i){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*cloud_ptrs[i], *cloud_rgb);

    int nColor = i % (colors_.size()/3);
    double r = colors_[nColor*3]*255.0;
    double g = colors_[nColor*3+1]*255.0;
    double b = colors_[nColor*3+2]*255.0;
    for (size_t j = 0; j < cloud_rgb->points.size (); j++){
        cloud_rgb->points[j].r = r;
        cloud_rgb->points[j].g = g;
        cloud_rgb->points[j].b = b;
    }
    combined_cloud += *cloud_rgb;
  }

  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(combined_cloud, output);

  output.header.stamp = ros::Time(secs, nsecs);
  output.header.frame_id = cloud_frame;
  hull_cloud_pub_.publish(output);
}


void Pass::publishHullsAsMarkers(const std::string& cloud_frame,
                                 std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > cloud_ptrs,
                                 int secs, int nsecs){
  geometry_msgs::Point point;
  std_msgs::ColorRGBA point_color;
  visualization_msgs::Marker marker;

  // define markers
  marker.header.frame_id = cloud_frame;
  marker.header.stamp = ros::Time(secs, nsecs);
  marker.ns = "hull lines";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::LINE_LIST; //visualization_msgs::Marker::POINTS;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = 0;
  marker.pose.position.y = 0;
  marker.pose.position.z = 0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.03;
  marker.scale.y = 0.03;
  marker.scale.z = 0.03;
  marker.color.a = 1.0;

  for (size_t i = 0; i < cloud_ptrs.size (); i++){
    int nColor = i % (colors_.size()/3);
    double r = colors_[nColor*3]*255.0;
    double g = colors_[nColor*3+1]*255.0;
    double b = colors_[nColor*3+2]*255.0;

    for (size_t j = 1; j < cloud_ptrs[i]->points.size (); j++){
      point.x = cloud_ptrs[i]->points[j-1].x;
      point.y = cloud_ptrs[i]->points[j-1].y;
      point.z = cloud_ptrs[i]->points[j-1].z;
      point_color.r = r;
      point_color.g = g;
      point_color.b = b;
      point_color.a = 1.0;
      marker.colors.push_back(point_color);
      marker.points.push_back(point);

      //
      point.x = cloud_ptrs[i]->points[j].x;
      point.y = cloud_ptrs[i]->points[j].y;
      point.z = cloud_ptrs[i]->points[j].z;
      point_color.r = r;
      point_color.g = g;
      point_color.b = b;
      point_color.a = 1.0;
      marker.colors.push_back(point_color);
      marker.points.push_back(point);
    }

    // start to end line:
    point.x = cloud_ptrs[i]->points[0].x;
    point.y = cloud_ptrs[i]->points[0].y;
    point.z = cloud_ptrs[i]->points[0].z;
    point_color.r = r;
    point_color.g = g;
    point_color.b = b;
    point_color.a = 1.0;
    marker.colors.push_back(point_color);
    marker.points.push_back(point);

    point.x = cloud_ptrs[i]->points[ cloud_ptrs[i]->points.size()-1 ].x;
    point.y = cloud_ptrs[i]->points[ cloud_ptrs[i]->points.size()-1 ].y;
    point.z = cloud_ptrs[i]->points[ cloud_ptrs[i]->points.size()-1 ].z;
    point_color.r = r;
    point_color.g = g;
    point_color.b = b;
    point_color.a = 1.0;
    marker.colors.push_back(point_color);
    marker.points.push_back(point);
  }
  marker.frame_locked = true;
  hull_markers_pub_.publish(marker);
}

void Pass::publishHullsAsMarkerArray(const std::string& cloud_frame,
                                     std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > cloud_ptrs,
                                     int secs, int nsecs){
  geometry_msgs::Point point;
  std_msgs::ColorRGBA point_color;
  visualization_msgs::MarkerArray ma;

  for (size_t i = 0; i < cloud_ptrs.size (); i++){
    visualization_msgs::Marker marker;
    marker.header.frame_id = cloud_frame;
    marker.header.stamp = ros::Time(secs, nsecs);
    marker.ns = "hull_" + std::to_string(i);
    marker.id = i;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.03;
    marker.scale.y = 0.03;
    marker.scale.z = 0.03;
    marker.color.a = 1.0;

    const int nColor = i % (colors_.size()/3);
    const double r = colors_[nColor*3]*255.0;
    const double g = colors_[nColor*3+1]*255.0;
    const double b = colors_[nColor*3+2]*255.0;

    marker.points.reserve(cloud_ptrs[i]->points.size());
    marker.colors.reserve(cloud_ptrs[i]->points.size());
    for (size_t j = 1; j < cloud_ptrs[i]->points.size(); j++){
      point.x = cloud_ptrs[i]->points[j-1].x;
      point.y = cloud_ptrs[i]->points[j-1].y;
      point.z = cloud_ptrs[i]->points[j-1].z;
      point_color.r = r;
      point_color.g = g;
      point_color.b = b;
      point_color.a = 1.0;
      marker.colors.push_back(point_color);
      marker.points.push_back(point);

      point.x = cloud_ptrs[i]->points[j].x;
      point.y = cloud_ptrs[i]->points[j].y;
      point.z = cloud_ptrs[i]->points[j].z;
      point_color.r = r;
      point_color.g = g;
      point_color.b = b;
      point_color.a = 1.0;
      marker.colors.push_back(point_color);
      marker.points.push_back(point);
    }

    // start to end line:
    point.x = cloud_ptrs[i]->points[0].x;
    point.y = cloud_ptrs[i]->points[0].y;
    point.z = cloud_ptrs[i]->points[0].z;
    point_color.r = r;
    point_color.g = g;
    point_color.b = b;
    point_color.a = 1.0;
    marker.colors.push_back(point_color);
    marker.points.push_back(point);

    point.x = cloud_ptrs[i]->points[ cloud_ptrs[i]->points.size()-1 ].x;
    point.y = cloud_ptrs[i]->points[ cloud_ptrs[i]->points.size()-1 ].y;
    point.z = cloud_ptrs[i]->points[ cloud_ptrs[i]->points.size()-1 ].z;
    point_color.r = r;
    point_color.g = g;
    point_color.b = b;
    point_color.a = 1.0;
    marker.colors.push_back(point_color);
    marker.points.push_back(point);

    marker.frame_locked = true;
    ma.markers.push_back(marker);
  }
  hull_marker_array_pub_.publish(ma);
}

int main( int argc, char** argv ){
  // Turn off warning message about labels
  // TODO: look into how labels are used
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);


  ros::init(argc, argv, "plane_seg_n3");
  ros::NodeHandle nh("~");
  Pass pass_pt (nh);
  std::unique_ptr<Pass> app = std::make_unique<Pass>(nh);
  ros::Rate rate(10);

  ROS_INFO_STREAM("ros node 3 ready");
  ROS_INFO_STREAM("=============================");
  ROS_INFO_STREAM("Waiting for ROS messages");

  // while (ros::ok()) {
  //   ros::spinOnce();    
  //   rate.sleep();  
  // }

  ros::spin();
  // ros::spinOnce();  

  return 1;
}
