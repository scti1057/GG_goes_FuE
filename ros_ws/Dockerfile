ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    locales sudo git wget curl nano \
    build-essential \
    python3-pip python3-venv \
    python3-colcon-common-extensions \
    python3-rosdep python3-vcstool \
    python3-numpy python3-opencv python3-torch python3-tqdm \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-rviz2 \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

COPY ./dds_profile.xml /home/${USERNAME}/dds_profile.xml
RUN chown ${USERNAME}:${USERNAME} /home/${USERNAME}/dds_profile.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/home/${USERNAME}/dds_profile.xml

WORKDIR /home/ros_ws
COPY ./src /home/ros_ws/src

RUN if [ ! -e /etc/ros/rosdep/sources.list.d/20-default.list ]; then rosdep init; fi
RUN rosdep update
RUN apt-get update && rosdep install --rosdistro ${ROS_DISTRO} --ignore-src --from-paths /home/ros_ws/src -r -y \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc \
    && echo "if [ -f /home/ros_ws/install/setup.bash ]; then source /home/ros_ws/install/setup.bash; fi" >> /etc/bash.bashrc

RUN chown -R ${USERNAME}:${USERNAME} /home/ros_ws

USER ${USERNAME}
WORKDIR /home/ros_ws

CMD ["bash"]
