#ifndef CARTESIAN_POSE_PASSTHROUGH_CONTROLLER_HPP_
#define CARTESIAN_POSE_PASSTHROUGH_CONTROLLER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "cartesian_pose_passthrough_controller/Utility.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "controller_interface/controller_interface.hpp"
#include <geometry_msgs/msg/pose.hpp>
#include <realtime_tools/realtime_buffer.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <cartesian_passthrough_msgs/action/follow_cartesian_trajectory.hpp>


namespace cartesian_pose_passthrough_controller
{

class CartesianPosePassthroughController : public controller_interface::ControllerInterface
{
public:
  CartesianPosePassthroughController();
  virtual ~CartesianPosePassthroughController() = default;

  // Lifecycle management
  controller_interface::CallbackReturn on_init() override;
  
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  // Main control loop
  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  // Interface configuration - override to use TCP pose interfaces
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

protected:

private:
  // Command action server
  using FollowCartesianTrajectory = cartesian_passthrough_msgs::action::FollowCartesianTrajectory;
  rclcpp_action::Server<FollowCartesianTrajectory>::SharedPtr action_server_;

  // Action server methods
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const FollowCartesianTrajectory::Goal> goal);
  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<FollowCartesianTrajectory>> goal_handle);
  void handle_accepted(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<FollowCartesianTrajectory>> goal_handle);
  void execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<FollowCartesianTrajectory>> goal_handle);
  
  // Interface naming
  std::string tcp_pose_interface_name_;
  
  // Helper methods
  void resetCommands();
  void updatePoseCommand(const geometry_msgs::msg::Pose::SharedPtr & msg);
  void commandCallback(const geometry_msgs::msg::Pose::SharedPtr msg);
};

} // namespace cartesian_pose_passthrough_controller

#endif // CARTESIAN_POSE_PASSTHROUGH_CONTROLLER_HPP_