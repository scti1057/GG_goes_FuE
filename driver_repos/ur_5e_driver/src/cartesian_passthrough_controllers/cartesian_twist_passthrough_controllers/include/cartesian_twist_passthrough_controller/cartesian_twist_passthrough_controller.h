#ifndef CARTESIAN_TWIST_PASSTHROUGH_CONTROLLER_HPP_
#define CARTESIAN_TWIST_PASSTHROUGH_CONTROLLER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "cartesian_twist_passthrough_controller/Utility.h"
#include "controller_interface/controller_interface.hpp"
#include <geometry_msgs/msg/twist.hpp>
#include <realtime_tools/realtime_buffer.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp>
#include <rclcpp_lifecycle/state.hpp>

namespace cartesian_twist_passthrough_controller
{

class CartesianTwistPassthroughController : public controller_interface::ControllerInterface
{
public:
  CartesianTwistPassthroughController();
  virtual ~CartesianTwistPassthroughController() = default;

  // Lifecycle management
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_init() override;
  
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  // Main control loop
  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  // Interface configuration - override to use TCP velocity interfaces
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

protected:

private:
  // Command subscriber for direct twist commands
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_subscriber_;
  realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::Twist>> rt_command_ptr_;
  
  // Timeout handling
  rclcpp::Time last_command_time_;
  double timeout_;
  bool command_timeout_;
  
  // Current commanded twist (for passthrough mode)
  ctrl::Vector6D current_twist_command_;
  
  // Interface naming
  std::string tcp_interface_name_;
  
  // Helper methods
  void resetCommands();
  bool checkTimeout(const rclcpp::Time & time);
  void updateTwistCommand(const geometry_msgs::msg::Twist::SharedPtr & msg);
  void commandCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
};

} // namespace cartesian_twist_passthrough_controller

#endif // CARTESIAN_TWIST_PASSTHROUGH_CONTROLLER_HPP_