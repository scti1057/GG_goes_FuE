ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

# Define user and home directory
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000
ARG WORKDIR=/home/$USERNAME

# Create a non-root user with the specified UID/GID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set ROS2 DDS profile
COPY ./dds_profile.xml /home/${USERNAME}
RUN chown ${USERNAME}:${USERNAME} /home/${USERNAME}/dds_profile.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/home/${USERNAME}/dds_profile.xml

# Set working directory and change ownership
WORKDIR $WORKDIR
RUN chown -R $USERNAME:$USERNAME $WORKDIR

# Install prerequisites
RUN sudo apt-get update && apt-get install -y git wget \
    g++ \
    ros-${ROS_DISTRO}-librealsense2* python3-rosdep

# Switch to non-root user
USER $USERNAME

# Create ROS2 workspace
RUN mkdir -p /home/$USERNAME/ros2_ws/src
WORKDIR /home/$USERNAME/ros2_ws/src

# Clone RealSense ROS2 driver repository
RUN git clone https://github.com/realsenseai/realsense-ros.git -b 4.55.1
WORKDIR /home/$USERNAME/ros2_ws/

RUN rosdep update
RUN rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y

RUN bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --symlink-install"

# Source ROS environment and set up entrypoint
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
RUN echo "source /home/$USERNAME/ros2_ws/install/setup.bash" >> ~/.bashrc

# Copy volumes into the container
COPY ./startup_scripts $WORKDIR/ros2_ws/startup_scripts

WORKDIR /home/${USERNAME}/ros2_ws/

CMD ["bash"]