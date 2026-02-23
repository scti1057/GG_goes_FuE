#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/header.hpp"
//#include "irp_ur5e_support/action/"type in the relevant ".hpp" if needed 
// till here everything incuded file is converted 
#include <array>
#include <string>
#include <chrono>

using namespace std::chrono_literals;
// chrono To specify time conveniently in seconds, milliseconds, microseconds,
// C++14 introduced chrono literals:
// Function to create and return a PoseStamped message
geometry_msgs::msg::PoseStamped get_pose_msg(float x, float y, float z, float q1, float q2, float q3, float q4,std::string frame)
{
    geometry_msgs::msg::PoseStamped posestamped_;
    geometry_msgs::msg::Point position_;
    geometry_msgs::msg::Quaternion orientation_;
    std_msgs::msg::Header header_;
    std_msgs::msg::String frame_id_;

    frame_id_.data = frame;

    position_.x = x;
    position_.y = y;
    position_.z = z;

    orientation_.x = q1;
    orientation_.y = q2;
    orientation_.z = q3;
    orientation_.w = q4;

    posestamped_.pose.position = position_;
    posestamped_.pose.orientation = orientation_;
    posestamped_.header.frame_id = frame_id_.data;

    return posestamped_;
}

// Function to print pose info, Syntax vise it would be the saem, I guess applying alais indicution would be better
void print_screen_info(const geometry_msgs::msg::PoseStamped &posestamped)
{
    RCLCPP_INFO(rclcpp::get_logger("pose_publisher"), // ros2 equivalent of ROS_INFO
        "moving to \n position: [%f, %f, %f] \n orientation: [%f, %f, %f, %f]",
        posestamped.pose.position.x, posestamped.pose.position.y, posestamped.pose.position.z,
        posestamped.pose.orientation.x, posestamped.pose.orientation.y, posestamped.pose.orientation.z,
        posestamped.pose.orientation.w);
}

int main(int argc, char **argv)
{
    // Initialize ROS 2
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("my_pose_publisher");

    // publisher (queue size 10)
    auto pose_pub = node->create_publisher<geometry_msgs::msg::PoseStamped>("target_frame", 10);

    // Define poses
    auto posestamped_1 = get_pose_msg(0.4, -0.1, 0.4, 0.0, 0.0, 0.0, 1.0, "base_link");
    auto posestamped_2 = get_pose_msg(0.4, -0.1, 0.6, 0.0, 0.0, 0.0, 1.0, "base_link");
    auto posestamped_3 = get_pose_msg(0.4, 0.3, 0.6, 0.0, 0.0, 0.0, 1.0, "base_link");
    auto posestamped_4 = get_pose_msg(0.4, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0, "base_link");

    std::array<geometry_msgs::msg::PoseStamped, 4> traj = {posestamped_1, posestamped_2, posestamped_3, posestamped_4};
    size_t i = 0;

    rclcpp::Rate loop_rate(0.333); // 3 seconds

    // Main publishing loop
    while (rclcpp::ok())
    {
        auto posestamped = traj[i];
        print_screen_info(posestamped);
        pose_pub->publish(posestamped);
        rclcpp::spin_some(node);
        loop_rate.sleep();

        i++;
        i %= traj.size();
    }

    rclcpp::shutdown();
    // interesting fact about shutdown application : Your program may exit, but some ROS 2 internal resources could remain allocated temporarily.
    // In longer-running applications or complex systems, this could lead to memory leaks or dangling DDS connections.
    return 0;
}
