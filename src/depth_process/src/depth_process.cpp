#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class DepthImageTester
{
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber depth_sub_;

public:
    DepthImageTester() : it_(nh_)
    {
        // 订阅深度图话题，请根据实际话题名称修改
        depth_sub_ = it_.subscribe("/camera/forward_depth", 1,
                                   &DepthImageTester::depthCallback, this);                           
        cv::namedWindow("Depth Image");
    }

    ~DepthImageTester()
    {
        cv::destroyWindow("Depth Image");
    }

    void depthCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        try
        {
            // 将ROS图像消息转换为OpenCV图像
            cv_bridge::CvImagePtr cv_ptr;
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
            
            // 对深度图像进行归一化以便显示（假设有效深度在0.5-6米）
            cv::Mat depth_display;
            double min_val = 0.5, max_val = 6.0;
            cv_ptr->image.convertTo(depth_display, CV_8UC1, 255.0/(max_val-min_val), -255.0*min_val/(max_val-min_val));
            
            // 应用颜色映射
            cv::Mat depth_colored;
            cv::applyColorMap(depth_display, depth_colored, cv::COLORMAP_JET);
            
            // 显示图像
            cv::imshow("Depth Image", depth_colored);
            cv::waitKey(1);

            // 打印基本信息
            ROS_INFO_STREAM("Received depth image: width=" << msg->width 
                           << ", height=" << msg->height 
                           << ", encoding=" << msg->encoding);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_unitree_depth");
    DepthImageTester tester;
    ros::spin();
    return 0;
}