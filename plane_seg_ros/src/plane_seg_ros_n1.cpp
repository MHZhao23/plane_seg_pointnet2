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

    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);

    void preProcessFromFile(std::string sceneName);
    void preProcessCloud(const std::string& cloudFrame, planeseg::LabeledCloud::Ptr& inCloud, Eigen::Vector3f origin, Eigen::Vector3f lookDir);

  private:
    ros::NodeHandle node_;
    std::vector<double> colors_;

    ros::Subscriber point_cloud_sub_;
    ros::Publisher preprocessed_cloud_pub, received_cloud_pub_, time_pub;

    std::string fixed_frame_ = "odom";  // Frame in which all results are published. "odom" for backwards-compatibility. Likely should be "map".

    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformListener tfListener_;
};

Pass::Pass(ros::NodeHandle node_):
    node_(node_),
    tfBuffer_(ros::Duration(5.0)),
    tfListener_(tfBuffer_) {

  // subscribe the input point cloud
  point_cloud_sub_ = node_.subscribe ("/ouster/points", 100,
                                   &Pass::pointCloudCallback, this);

  // publishing the pre-processed point cloud from pointnet2
  time_pub = node_.advertise<std_msgs::Time>("/plane_seg_n1/start_time", 100);
  received_cloud_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/plane_seg_n1/received_cloud", 100);
  preprocessed_cloud_pub = node_.advertise<sensor_msgs::PointCloud2>("/plane_seg_n1/preprocessed_cloud", 100);

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

// process a point cloud 
// This method is mostly for testing
// To transmit a static point cloud:
// rosrun pcl_ros pcd_to_pointcloud 06.pcd   _frame_id:=/odom /cloud_pcd:=/plane_seg/point_cloud_in
void Pass::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg){
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

  preProcessCloud(msg->header.frame_id, inCloud, origin, lookDir);
}


void Pass::preProcessFromFile(std::string sceneName){

  // to allow ros connections to register
  sleep(2);

  std::string home_dir = ros::package::getPath("plane_seg_ros");
  std::string inFile = home_dir + "/data/" + sceneName + ".pcd";
  Eigen::Vector3f origin, lookDir;
  origin << 0, 0, 0;
  lookDir << 1, 0, 0;

  std::cout << "\n =========== Processing test example " << sceneName << " ===========\n";
  std::cout << inFile << "\n";

  std::size_t found_pcd = inFile.find(".pcd");

  planeseg::LabeledCloud::Ptr inCloud(new planeseg::LabeledCloud());
  if (found_pcd!=std::string::npos){
    std::cout << "readpcd\n";
    pcl::io::loadPCDFile(inFile, *inCloud);
  }else{
    std::cout << "extension not understood\n";
    return;
  }

  preProcessCloud(fixed_frame_, inCloud, origin, lookDir);
}


void Pass::preProcessCloud(const std::string& cloudFrame, planeseg::LabeledCloud::Ptr& inCloud, Eigen::Vector3f origin, Eigen::Vector3f lookDir){
  ros::Time begin = ros::Time::now();
  // std::chrono::time_point<std::chrono::high_resolution_clock> startT;
  // startT = std::chrono::high_resolution_clock::now();

  if (time_pub.getNumSubscribers() > 0) {
    std_msgs::Time timeMsg;
    timeMsg.data = begin;
    time_pub.publish(timeMsg);
  }

  if (received_cloud_pub_.getNumSubscribers() > 0) {
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*inCloud, output);
    output.header.stamp = ros::Time(0, 0);
    output.header.frame_id = cloudFrame;
    received_cloud_pub_.publish(output);
  }

  planeseg::Preprocessing preprossesor;
  preprossesor.setSensorPose(origin, lookDir);
  preprossesor.setFrame(cloudFrame);
  preprossesor.setCloud(inCloud);
  preprossesor.setDebug(false);
  planeseg::LabeledCloud::Ptr result_cloud = preprossesor.go();
            
  if (preprocessed_cloud_pub.getNumSubscribers() > 0) {
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*result_cloud, output);
    output.header.stamp = ros::Time(0, 0);
    output.header.frame_id = cloudFrame;
    preprocessed_cloud_pub.publish(output);
  }
}


int main( int argc, char** argv ){
  // Turn off warning message about labels
  // TODO: look into how labels are used
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);


  ros::init(argc, argv, "plane_seg_n1");
  ros::NodeHandle nh("~");
  Pass pass_pt (nh);
  std::unique_ptr<Pass> app = std::make_unique<Pass>(nh);
  ros::Rate rate(10);

  ROS_INFO_STREAM("ros node 1 ready");
  ROS_INFO_STREAM("=============================");

  std::string scene;
  // app->preProcessFromFile("realdata/scene1_12");
  // app->preProcessFromFile("realdata/scene3_15");
  // app->preProcessFromFile("realdata/scene6_21");
  app->preProcessFromFile("realdata/scene7_9");
  // app->preProcessFromFile("realdata/scene8_4");
  // app->preProcessFromFile("realdata/testscene25_19");
  // app->preProcessFromFile("realdata/steps_2");
  // app->preProcessFromFile("realdata/terrain_0");
  // app->preProcessFromFile("benchmark/rgbd_dataset_freiburg3_long_office_household_validation/1341848149.066840");
  ros::spin();

  return 1;
}
