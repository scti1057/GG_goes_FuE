# Universal Robot Driver #
Dockerized Driver for the UR cobot series. Functionality is based on the offical packages provided by UR ([Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver) and [Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)).

## Packages provided within this Docker ##
- `ur_description` - URDF and configs for all UR-Robots
- `ur_driver` - Driver package for all UR-Robots
- `Cartesian ROS Controllers` - Additional driver package for cartesian control

## Package you need to provide locally
- `ur_cell_description` - robot and cell description (custom URDF) and custom robot calibration as well as needed stls, meshes and urdfs for the custom cell

This file need to be mounted into the robot driver container within the project specific docker compose file.

## Usage

### Manual standalone usage
You can run the container using docker run and call one of the predefined bringup launchfiles provided by the [ur_robot_driver](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/tree/master) package. 
Or define a custom robot description containing the entire robot cell. For an example take a look the [workspace repo for the UR5e cell](https://irp-nas.fox-stairs.ts.net/Workspaces/ur_5e).

### Autostartup of driver using a Docker Compose script
Within your Docker Compose script add:
```yaml
ur_driver:
    image: irp-nas.fox-stairs.ts.net/hardware_drivers/ur_driver:humble
    container_name: ur_driver
    environment:
      - ROBOT_IP="192.168.1.100"
    volumes:
      - ./deps/ur_cell_description:/home/ros_ws/src/Universal_Robots_ROS2_Description  # Mount local folder overwriting the standard description package
      # Mount the needed controller config file
      - ./deps/ros2_control_configs/ur_5e_axia_controller.yaml:/home/ros_ws/src/Universal_Robots_ROS2_Driver/ur_robot_driver/config/ur_controllers.yaml 

    network_mode: host
    entrypoint: ["/startup_scripts/ur5e_entrypoint.sh"] # Choose your needed startup script 

    tty: true
    stdin_open: true
```

The [startup script](/startup_scripts/ur5e_entrypoint.sh) takes care of starting the needed ROS Nodes on the client side. The standard startup_script starts the driver for the UR5e with a custom launchfile for the robot cell.
If you want to be able to start another robot or another common robot config feel free to add another startup_script.

## Making changes to the driver container
If you need to make changes to this driver container clone the repo locally and run:
```shell
 docker build -t <image_name>:<image_tag> .
 ```
 within the root folder of the repo.

 In case you want to push the build image to the IRP container registry stick to the following syntax:
 ```shell
 docker build -t irp-nas.fox-stairs.ts.net/hardware_drivers/ur_driver:<tag> .
 ```    
 After that you can push the image as a package using:
 ```shell
 docker push irp-nas.fox-stairs.ts.net/hardware_drivers/ur_driver:<tag>
```
