#include "cartesian_pose_passthrough_controller/cartesian_pose_passthrough_controller.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cartesian_pose_passthrough_controller
{

CartesianPosePassthroughController::CartesianPosePassthroughController()
: controller_interface::ControllerInterface()
{
  current_pose_command_.setZero();
}

controller_interface::InterfaceConfiguration 
CartesianPosePassthroughController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  
  // Build interface names - these must match your hardware interface exactly
  config.names.push_back(tcp_pose_interface_name_ + "/position.x");
  config.names.push_back(tcp_pose_interface_name_ + "/position.y");
  config.names.push_back(tcp_pose_interface_name_ + "/position.z");
  config.names.push_back(tcp_pose_interface_name_ + "/orientation.x");
  config.names.push_back(tcp_pose_interface_name_ + "/orientation.y");
  config.names.push_back(tcp_pose_interface_name_ + "/orientation.z");

  return config;
}

controller_interface::InterfaceConfiguration 
CartesianPosePassthroughController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  
  // Optional: Read TCP velocity state interfaces for feedback
  // config.names.push_back(tcp_pose_interface_name_ + "/position.x");
  // config.names.push_back(tcp_pose_interface_name_ + "/position.y");
  // config.names.push_back(tcp_pose_interface_name_ + "/position.z");
  // config.names.push_back(tcp_pose_interface_name_ + "/orientation.x");
  // config.names.push_back(tcp_pose_interface_name_ + "/orientation.y");
  // config.names.push_back(tcp_pose_interface_name_ + "/orientation.z");

  return config;
}

controller_interface::CallbackReturn CartesianPosePassthroughController::on_init()
{
 try {
     // Require parameters to be provided externally (no default values here).
     // Do NOT declare them here — declaring would make them "present" with a default.
     auto result = get_node()->list_parameters({"tcp_pose_interface_name"}, 10);
     bool has_tcp = std::find(result.names.begin(), result.names.end(), "tcp_pose_interface_name") != result.names.end();
     if (!has_tcp) {
       RCLCPP_ERROR(get_node()->get_logger(),
                    "Required parameter(s) missing: tcp_pose_interface_name=%s",
                    has_tcp ? "present" : "missing");
       return controller_interface::CallbackReturn::ERROR;
     }
     // Read parameters (they exist because controller_manager should set them from YAML)
     get_node()->get_parameter("tcp_pose_interface_name", tcp_pose_interface_name_);


     // Create action server
     action_server_ = rclcpp_action::create_server<FollowCartesianTrajectory>(
       get_node(),
       "follow_cartesian_trajectory",
       std::bind(&CartesianPosePassthroughController::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
       std::bind(&CartesianPosePassthroughController::handle_cancel, this, std::placeholders::_1),
       std::bind(&CartesianPosePassthroughController::handle_accepted, this, std::placeholders::_1));

  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception during init: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn CartesianPosePassthroughController::on_configure(
  const rclcpp_lifecycle::State & previous_state)
{
  // Get parameters
  tcp_pose_interface_name_ = get_node()->get_parameter("tcp_pose_interface_name").as_string();
  
  // Create pose command subscriber
  pose_subscriber_ = get_node()->create_subscription<geometry_msgs::msg::Pose>(
  "~/cmd_pose",
  rclcpp::QoS(1).reliable(),
  std::bind(&CartesianPosePassthroughController::commandCallback, this, std::placeholders::_1));
  
  RCLCPP_INFO(get_node()->get_logger(), 
              "Cartesian Pose Passthrough Controller configured with interface: %s", 
              tcp_pose_interface_name_.c_str());
  
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn CartesianPosePassthroughController::on_activate(
  const rclcpp_lifecycle::State & previous_state)
{
  
  // Verify we have the expected interfaces
  if (command_interfaces_.size() != 6) {
    RCLCPP_ERROR(get_node()->get_logger(), 
                 "Expected 6 command interfaces, got %zu", command_interfaces_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (state_interfaces_.size() != 6) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "Expected 6 state interfaces, got %zu", state_interfaces_.size());
    return controller_interface::CallbackReturn::ERROR;
  }
  
  RCLCPP_INFO(get_node()->get_logger(), "Cartesian Pose Passthrough Controller activated");
  
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn CartesianPosePassthroughController::on_deactivate(
  const rclcpp_lifecycle::State & previous_state)
{
  RCLCPP_INFO(get_node()->get_logger(), "Cartesian Pose Passthrough Controller deactivated");

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type CartesianPosePassthroughController::update(
  const rclcpp::Time & time, const rclcpp::Duration & period)
{
  // Get latest command and update current pose
  auto pose_command = rt_command_ptr_.readFromRT();
  if (pose_command && (*pose_command)) {
    updatePoseCommand(*pose_command);

    // Ensure we have the correct number of command interfaces (should be 6)
    if (command_interfaces_.size() != 6) {
      RCLCPP_ERROR(get_node()->get_logger(), 
                  "Expected 6 command interfaces, got %zu", command_interfaces_.size());
      return controller_interface::return_type::ERROR;
    }
    
    // Set the TCP pose commands directly to hardware interfaces
    // Order: position.x, position.y, position.z, orientation.x, orientation.y, orientation.z
    command_interfaces_[0].set_value(current_pose_command_[0]); // position.x
    command_interfaces_[1].set_value(current_pose_command_[1]); // position.y
    command_interfaces_[2].set_value(current_pose_command_[2]); // position.z
    command_interfaces_[3].set_value(current_pose_command_[3]); // orientation.x
    command_interfaces_[4].set_value(current_pose_command_[4]); // orientation.y
    command_interfaces_[5].set_value(current_pose_command_[5]); // orientation.z
    
  }
  return controller_interface::return_type::OK;
}

rclcpp_action::GoalResponse handle_goal(
  const rclcpp_action::GoalUUID & uuid,
  std::shared_ptr<const FollowCartesianTrajectory::Goal> goal)
{
  RCLCPP_INFO(this->get_logger(), "Received goal request with order %d", goal->order);
  (void)uuid;
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse handle_cancel(
  const std::shared_ptr<FollowCartesianTrajectory::GoalHandle> goal_handle)
{
  RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
  (void)goal_handle;
  return rclcpp_action::CancelResponse::ACCEPT;
}

void handle_accepted(const std::shared_ptr<GoalHandleFollowCartesianTrajectory> goal_handle)
{
  // This needs to return quickly to avoid blocking the executor, so spin up a new thread
  std::thread{std::bind(&CartesianPosePassthroughController::execute, this, std::placeholders::_1), goal_handle}.detach();

} 

void execute(const std::shared_ptr<GoalHandleFollowCartesianTrajectory> goal_handle)
{
  RCLCPP_INFO(this->get_logger(), "Executing goal");
  // Do lots of stuff here

  // After finishing the goal, set the result
  auto result = std::make_shared<FollowCartesianTrajectory::Result>();
  result->error_code = FollowCartesianTrajectory::Result::SUCCESSFUL;
  goal_handle->succeed(result);
}

}

// namespace cartesian_pose_passthrough_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  cartesian_pose_passthrough_controller::CartesianPosePassthroughController, 
  controller_interface::ControllerInterface)