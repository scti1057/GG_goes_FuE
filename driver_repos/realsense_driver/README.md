# Realsense ROS2 driver
Container used to run the ROS2 driver for the Intel Realsense cameras.

## Auto deployment with suggested settings using Docker Compose
The current containter holds an [autostart](/startup_scripts/realsense_entrypoint.sh) file which start the needed ROS2 launch file.
You can do this by using the following syntax within your docker compose file:
```yaml
realsense_driver:
  image: irp-nas.fox-stairs.ts.net/hardware_drivers/realsense_driver:humble
  container_name: camera_driver
  privileged: true
  network_mode: host
  environment:
    - DISPLAY=${DISPLAY}
    - ROS_DOMAIN_ID=0
    - PYTHONPATH=${PYTHONPATH}
    - QT_X11_NO_MITSHM=1
    - XAUTHORITY=${XAUTHORITY}
  volumes:
    - ${XAUTHORITY}:${XAUTHORITY}:rw
    - /tmp/.X11-unix:/tmp/.X11-unix
    - /dev:/dev
  entrypoint: ["/home/ros/ros2_ws/startup_scripts/realsense_entrypoint_filter.sh"]
  tty: true
  stdin_open: true
```

## Manual usage
Clone this repo and use the provided **build_docker.sh** script to build the container.
Next use the provided **start_docker.sh** script to start the container. This script starts the container, mounts the needed directories.\
Now from inside the docker start the launchscript provided by Intel by running
```shell
ros2 launch realsense2_camera rs_launch.py align_depth:=true
```
You can pass different parameters to the launchfile using the ROS2 launchfile params. The available parameters can be found here: https://github.com/realsenseai/realsense-ros?tab=readme-ov-file#parameters

## Making changes to the driver container
If you need to make changes to this driver container clone the repo locally and run:

```shell
docker build -t <image_name>:<image_tag> .
```
within the root folder of the repo.

In case you want to push the build image to the IRP container registry stick to the following syntax:
```shell
docker build -t irp-nas.fox-stairs.ts.net/hardware_drivers/realsense_driver:<tag> .
```
After that you can push the image as a package using:
```shell
docker push irp-nas.fox-stairs.ts.net/hardware_drivers/realsense_driver:<tag>
```
