# Usage: source /home/duckie5/Documents/GG_goes_FundE/ros_ws/scripts/build_and_source.sh

# Ensure we are sourced (works in bash + zsh)
if ! (return 0 2>/dev/null); then
  echo "Please run: source scripts/build_and_source.sh"
  exit 1
fi

WS_DIR="/home/duckie5/Documents/GG_goes_FundE/ros_ws"

# Basic sanity checks
if [ ! -d "$WS_DIR" ]; then
  echo "WS_DIR not found: $WS_DIR"
  return 1
fi
if [ ! -f "/opt/ros/humble/setup.bash" ]; then
  echo "ROS2 Humble not found at /opt/ros/humble"
  return 1
fi
if [ ! -f "$WS_DIR/.venv/bin/activate" ]; then
  echo "venv not found: $WS_DIR/.venv"
  return 1
fi

# Clean noisy vars (optional)
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH 2>/dev/null
unset CATKIN_INSTALL_INTO_PREFIX_ROOT CATKIN_SYMLINK_INSTALL 2>/dev/null

# Source underlay + venv
source /opt/ros/humble/setup.bash || return 1
source "$WS_DIR/.venv/bin/activate" || return 1

cd "$WS_DIR" || return 1

# Clean build (most robust)
rm -rf build install log

# Build all packages (merged install)
colcon build --symlink-install --merge-install || return 1

# Source overlay
source "$WS_DIR/install/setup.bash" || return 1

echo "✅ Built & sourced workspace: $WS_DIR"