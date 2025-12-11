// 在 src/depth_test.cpp 中写入
#include <ros/ros.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "depth_test_node");
    ROS_INFO("Depth test node started!");
    ros::spin();
    return 0;
}