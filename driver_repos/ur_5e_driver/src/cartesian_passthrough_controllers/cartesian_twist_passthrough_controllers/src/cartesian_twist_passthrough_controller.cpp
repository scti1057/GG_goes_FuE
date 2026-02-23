#include "cartesian_twist_passthrough_controller/cartesian_twist_passthrough_controller.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cartesian_twist_passthrough_controller
{

CartesianTwistPassthroughController::CartesianTwistPassthroughController()
: controller_interface::ControllerInterface()
{
  current_twist_command_.setZero();
}

controller_interface::InterfaceConfiguration 
CartesianTwistPassthroughController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  
  // Build interface names - these must match your hardware interface exactly
  config.names.push_back(tcp_interface_name_ + "/linear.x");
  config.names.push_back(tcp_interface_name_ + "/linear.y");
  config.names.push_back(tcp_interface_name_ + "/linear.z");
  config.names.push_back(tcp_interface_name_ + "/angular.x");
  config.names.push_back(tcp_interface_name_ + "/angular.y");
  config.names.push_back(tcp_interface_name_ + "/angular.z");
  
  return config;
}

controller_interface::InterfaceConfiguration 
CartesianTwistPassthroughController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  
  // Optional: Read TCP velocity state interfaces for feedback
  config.names.push_back(tcp_interface_name_ + "/linear.x");
  config.names.push_back(tcp_interface_name_ + "/linear.y");
  config.names.push_back(tcp_interface_name_ + "/linear.z");
  config.names.push_back(tcp_interface_name_ + "/angular.x");
  config.names.push_back(tcp_interface_name_ + "/angular.y");
  config.names.push_back(tcp_interface_name_ + "/angular.z");
  
  return config;
}

controller_interface::CallbackReturn CartesianTwistPassthroughController::on_init()
{
 try {
     // Require parameters to be provided externally (no default values here).
     // Do NOT declare them here — declaring would make them "present" with a default.
     auto result = get_node()->list_parameters({"tcp_interface_name", "timeout"}, 10);
     bool has_tcp = std::find(result.names.begin(), result.names.end(), "tcp_interface_name") != result.names.end();
     bool has_timeout = std::find(result.names.begin(), result.names.end(), "timeout") != result.names.end();
     if (!has_tcp || !has_timeout) {
       RCLCPP_ERROR(get_node()->get_logger(),
                    "Required parameter(s) missing: tcp_interface_name=%s timeout=%s",
                    has_tcp ? "present" : "missing",
                    has_timeout ? "present" : "missing");
       return controller_interface::CallbackReturn::ERROR;
     }
     // Read parameters (they exist because controller_manager should set them from YAML)
     get_node()->get_parameter("tcp_interface_name", tcp_interface_name_);
     get_node()->get_parameter("timeout", timeout_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception during init: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn CartesianTwistPassthroughController::on_configure(
  const rclcpp_lifecycle::State & previous_state)
{
  // Get parameters
  tcp_interface_name_ = get_node()->get_parameter("tcp_interface_name").as_string();
  timeout_ = get_node()->get_parameter("timeout").as_double();
  
  // Create twist command subscriber
  twist_subscriber_ = get_node()->create_subscription<geometry_msgs::msg::Twist>(
  "~/cmd_vel",
  rclcpp::QoS(1).reliable(),
  std::bind(&CartesianTwistPassthroughController::commandCallback, this, std::placeholders::_1));
  
  RCLCPP_INFO(get_node()->get_logger(), 
              "Cartesian Twist Controller configured with interface: %s", 
              tcp_interface_name_.c_str());
  
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn CartesianTwistPassthroughController::on_activate(
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
  
  RCLCPP_INFO(get_node()->get_logger(), "Cartesian Twist Controller activated");
  
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn CartesianTwistPassthroughController::on_deactivate(
  const rclcpp_lifecycle::State & previous_state)
{
  
  // Zero out hardware interfaces
  for (auto & command_interface : command_interfaces_) {
    command_interface.set_value(0.0);
  }
  
  RCLCPP_INFO(get_node()->get_logger(), "Cartesian Twist Controller deactivated");
  
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type CartesianTwistPassthroughController::update(
  const rclcpp::Time & time, const rclcpp::Duration & period)
{
  // Check for timeout
  if (checkTimeout(time)) {
    resetCommands();
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 5000,
      "TCP velocity command timeout - stopping robot");
  }
  
  // Get latest command and update current twist
  auto twist_command = rt_command_ptr_.readFromRT();
  if (!command_timeout_ && twist_command && (*twist_command)) {
    updateTwistCommand(*twist_command);
  }
  
  // Ensure we have the correct number of command interfaces (should be 6)
  if (command_interfaces_.size() != 6) {
    RCLCPP_ERROR(get_node()->get_logger(), 
                 "Expected 6 command interfaces, got %zu", command_interfaces_.size());
    return controller_interface::return_type::ERROR;
  }
  
  // Set the TCP velocity commands directly to hardware interfaces
  // Order: linear.x, linear.y, linear.z, angular.x, angular.y, angular.z
  command_interfaces_[0].set_value(current_twist_command_[0]); // linear.x
  command_interfaces_[1].set_value(current_twist_command_[1]); // linear.y
  command_interfaces_[2].set_value(current_twist_command_[2]); // linear.z
  command_interfaces_[3].set_value(current_twist_command_[3]); // angular.x
  command_interfaces_[4].set_value(current_twist_command_[4]); // angular.y
  command_interfaces_[5].set_value(current_twist_command_[5]); // angular.z
  
  return controller_interface::return_type::OK;
}

void CartesianTwistPassthroughController::resetCommands()
{
  rt_command_ptr_.reset();
  last_command_time_ = get_node()->get_clock()->now();
  command_timeout_ = true;
  current_twist_command_.setZero();
}

bool CartesianTwistPassthroughController::checkTimeout(const rclcpp::Time & time)
{
  if (timeout_ <= 0.0) {
    // No timeout configured
    return false;
  }
  if (!command_timeout_ && (time - last_command_time_).seconds() > timeout_) {
    command_timeout_ = true;
    current_twist_command_.setZero();
    return true;
  }
  return command_timeout_;
}

void CartesianTwistPassthroughController::updateTwistCommand(
  const geometry_msgs::msg::Twist::SharedPtr & msg)
{
  // Convert Twist message to our internal format
  current_twist_command_[0] = msg->linear.x;
  current_twist_command_[1] = msg->linear.y;
  current_twist_command_[2] = msg->linear.z;
  current_twist_command_[3] = msg->angular.x;
  current_twist_command_[4] = msg->angular.y;
  current_twist_command_[5] = msg->angular.z;
}

void CartesianTwistPassthroughController::commandCallback(
  const geometry_msgs::msg::Twist::SharedPtr msg)
{
  rt_command_ptr_.writeFromNonRT(msg);
  last_command_time_ = get_node()->get_clock()->now();
  command_timeout_ = false;
}

} // namespace cartesian_twist_passthrough_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  cartesian_twist_passthrough_controller::CartesianTwistPassthroughController, 
  controller_interface::ControllerInterface)