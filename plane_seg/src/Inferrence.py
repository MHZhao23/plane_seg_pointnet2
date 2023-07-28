#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2

def callback():
    rospy.loginfo(rospy.get_caller_id())

def listener():
    rospy.init_node('plane_seg_listener', anonymous=True)
    rospy.Subscriber("/plane_seg/processed_cloud", PointCloud2, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()