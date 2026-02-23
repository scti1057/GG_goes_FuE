FROM ros:humble-ros-core
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl nano iputils-ping \
    build-essential ros-humble-ament-cmake \
    python3-pip python3-colcon-common-extensions \
    python3-rosdep python3-vcstool \
    libyaml-cpp-dev \
    python3-dev \
    python3-setuptools \
    python3-numpy \
    libboost-all-dev \
    ros-${ROS_DISTRO}-moveit \
    ros-${ROS_DISTRO}-moveit-setup-assistant \
    ros-${ROS_DISTRO}-moveit-servo \
    ros-${ROS_DISTRO}-moveit-ros-visualization \
    && rm -rf /var/lib/apt/lists/*

# Set ROS2 DDS profile
COPY ./dds_profile.xml /root/
RUN chown root:root /root/dds_profile.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/root/dds_profile.xml

WORKDIR /home/ros_ws
RUN mkdir -p src
WORKDIR /home/ros_ws/src

# Copy all source files for the robot, sensor and related packages
COPY ./src/ /home/ros_ws/src/

WORKDIR /home/ros_ws
RUN rosdep init
RUN rosdep update
RUN apt-get update
RUN rosdep install --ignore-src --from-paths . -y -r

RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && colcon build --packages-skip cartesian_controller_simulation cartesian_controller_tests cartesian_passthrough_trajectory_controller cartesian_pose_passthrough_controller cartesian_passthrough_msgs cartesian_trajectory_interpolation"
RUN echo "source /home/ros_ws/install/setup.bash" >> ~/.bashrc

COPY ./startup_scripts /startup_scripts
RUN chmod +x /startup_scripts/*.sh

CMD ["bash"]
