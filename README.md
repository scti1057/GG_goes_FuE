# GG_goes_FuE - IBVS on UR5e with RealSense (ROS2 Humble)

## 1) Project Purpose
This project implements a modular image-based visual servoing (IBVS) stack for a UR5e robot with a RealSense camera.

Main goals:
- Build a robust visual reference from multiple frames (not one-shot).
- Match only against the saved reference features.
- Drive the robot via Cartesian twist commands until image error is small.
- Keep components modular so detectors, matching, filtering, and control can be swapped independently.

Current control setup:
- Translational control: `x, y, z`
- Rotational control: `wz`
- Disabled by design: `wx=0`, `wy=0`

## 2) Repository Layout
```
GG_goes_FuE/
├─ driver_repos/
│  ├─ realsense_driver/
│  └─ ur_5e_driver/
├─ ros_ws/
│  ├─ compose.yaml
│  ├─ scripts/
│  │  ├─ build_and_source.sh
│  │  └─ ibvs_session_manager.py
│  ├─ src/
│  │  ├─ ibvs_msgs/
│  │  ├─ ibvs_perception/
│  │  ├─ ibvs_reference/
│  │  ├─ ibvs_matching/
│  │  └─ ibvs_control/
│  └─ third_party/
└─ README.md
```

## 3) High-Level Architecture
Pipeline:
1. `ibvs_perception/keypoint` detects keypoints + descriptors on camera images.
2. `ibvs_reference/reference_manager` captures a robust reference set (time window + consistency counts).
3. `ibvs_matching/descriptor_matcher` matches live descriptors to reference descriptors.
4. `ibvs_matching/matches_viz` visualizes match tracks with fade logic.
5. `ibvs_control/ibvs_twist_controller` computes IBVS twist and publishes to UR twist controller.

Data flow:
```
/camera/camera/color/image_raw
        |
        v
   keypoint node  --> /ibvs/keypoints
                        |
                        +--> reference_manager --> /ibvs/reference/keypoints (latched/transient)
                        |                         /ibvs/init_done (latched/transient)
                        |
                        +--> descriptor_matcher --> /ibvs/matches
                                                       |
                                                       +--> matches_viz --> /ibvs/debug/matches_image
                                                       |
                                                       +--> ibvs_twist_controller --> /cartesian_twist_passthrough_controller/cmd_vel
```

## 4) Docker Architecture
Three containers are used:

- `camera_driver` (`driver_repos/realsense_driver`)
  - RealSense publish stack.
- `ros2_ur_driver` (`driver_repos/ur_5e_driver`)
  - UR ROS2 driver + controllers.
- `ros_ws` (`ros_ws/compose.yaml`)
  - IBVS workspace and custom nodes.

All use `network_mode: host` and must share the same `ROS_DOMAIN_ID`.

## 5) ROS Packages, Nodes, Topics

## 5.1 `ibvs_msgs`
Custom messages:

### `ibvs_msgs/msg/Keypoints.msg`
- `std_msgs/Header header`
- `float32[] xy` (flattened `[x0,y0,x1,y1,...]`)
- `uint32 descriptor_dim`
- `float32[] descriptors` (flattened `N*D`)
- `float32[] scores`

### `ibvs_msgs/msg/Matches.msg`
- `std_msgs/Header header`
- `uint32[] ref_id` (index into reference set)
- `float32[] xy` (current measurement positions)
- `float32[] sim` (cosine similarity)

## 5.2 `ibvs_perception` - `keypoint`
Node: `ibvs_perception.keypoint_node`

Purpose:
- Detect keypoints/descriptors with selectable detector.
- Optional depth-based ROI filtering.
- Publish debug overlays.

Subscriptions:
- `input_topic` (default `/camera/camera/color/image_raw`)
- `depth_topic` (default `/camera/camera/aligned_depth_to_color/image_raw`)

Publications:
- `keypoints_topic` (default `/ibvs/keypoints`)
- `output_topic` (default `/ibvs/debug/keypoints_image`)
- `binary_output_topic` (default `/ibvs/debug/near_mask`)

Detector options:
- `sift`, `orb`, `akaze`, `superpoint`, `aliked`, `xfeat`

Important params:
- `detector_type`, `device`, `top_k`, `max_num_keypoints`, `nfeatures`
- `use_depth_roi`, `depth_scale`, `min_valid_depth_m`, `near_mask_dilate_px`

## 5.3 `ibvs_reference` - `reference_manager`
Node: `ibvs_reference.reference_manager_node`

Purpose:
- Build robust reference over time window.
- Maintain candidate DB via descriptor + pixel gating.
- Select top-K stable features.
- Publish reference and init status as transient_local.

Subscriptions:
- `/ibvs/keypoints`
- camera image topic (default `/camera/camera/color/image_raw`) for overlay save

Service:
- `/ibvs/reference/start_capture` (`std_srvs/srv/Trigger`)

Publications:
- `/ibvs/init_done` (`std_msgs/Bool`, transient_local)
- `/ibvs/reference/keypoints` (`ibvs_msgs/Keypoints`, transient_local)
- optional debug image topic

Important params:
- `init_duration_sec`, `ref_top_k`
- `desc_match_threshold`, `max_px_dist`, `desc_ema_alpha`
- `reference_republish_hz`
- `save_overlay_path`, `save_npz_path`

## 5.4 `ibvs_matching` - `descriptor_matcher`
Node: `ibvs_matching.descriptor_matcher_node`

Purpose:
- Match live keypoints to cached reference descriptors.
- Publish only reference-indexed matches.

Subscriptions:
- `/ibvs/keypoints`
- `/ibvs/reference/keypoints` (transient_local QoS)

Publications:
- `/ibvs/matches`

Important params:
- `match_threshold`
- `mutual_check`

## 5.5 `ibvs_matching` - `matches_viz`
Node: `ibvs_matching.matches_viz_node`

Purpose:
- Visualize matched reference points with fade-out if temporarily lost.

Subscriptions:
- camera image topic
- `/ibvs/matches`

Publications:
- `/ibvs/debug/matches_image`

Important params:
- `miss_max`
- `radius`

## 5.6 `ibvs_control` - `ibvs_twist_controller`
Node: `ibvs_control.ibvs_twist_controller_node`

Purpose:
- Compute IBVS Cartesian twist from `matches + reference`.
- Enforce safety gates and stop criteria.
- Publish command twist for UR passthrough controller.

Subscriptions:
- `/ibvs/matches`
- `/ibvs/reference/keypoints` (transient_local)
- `/ibvs/init_done` (transient_local)

Publications:
- `/cartesian_twist_passthrough_controller/cmd_vel`
- `/ibvs/control/goal_reached` (transient_local)

Current default control params (tuned):
- `lambda_gain=0.12`
- `dls_damping=0.1`
- `min_matches=60`
- `error_stop_px=10.0`
- `stop_hold_sec=0.8`
- `max_linear_speed=0.004`
- `max_angular_speed=0.05`
- DOFs: `vx,vy,vz,wz=true`, `wx,wy=false`
- axis signs: `vx=-1`, `vy=+1`, `vz=-1`, `wz=-1`
- `enable_motion=false` by default (safety)

Notes:
- If camera mount or frame convention changes, axis signs likely need retuning.
- Keep `enable_motion=false` until system state and controllers are confirmed.

## 5.7 Session Tool - `ibvs_session_manager.py`
Script: `ros_ws/scripts/ibvs_session_manager.py`

Purpose:
- Interactive start/stop orchestration for:
  - `keypoint`
  - `reference_manager`
  - `descriptor_matcher`
  - `matches_viz`

Menu:
- `1` Core start
- `2` Initialization start (service call + wait for `/ibvs/init_done=true`)
- `3` Tracking start
- `4` Tracking stop
- `5` Status
- `6` Stop all
- `q` Quit + cleanup

Logs:
- `/tmp/ibvs_manager_logs`

## 6) Essential Commands

## 6.1 Build/start all containers
From host:
```bash
cd /home/wife1013/GG_goes_FuE/driver_repos/realsense_driver && docker compose up -d --build
cd /home/wife1013/GG_goes_FuE/driver_repos/ur_5e_driver && docker compose up -d --build
cd /home/wife1013/GG_goes_FuE/ros_ws && HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d --build
```

Check containers:
```bash
docker ps
```

Expected names:
- `camera_driver`
- `ros2_ur_driver`
- `ros_ws`

## 6.2 Build and source workspace (inside `ros_ws` container)
```bash
docker exec -it ros_ws bash -lc 'cd /home/ros_ws && source scripts/build_and_source.sh'
```

## 6.3 Start session manager
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && python3 /home/ros_ws/scripts/ibvs_session_manager.py --debug'
```

Typical flow in manager:
- `6` (clean stop)
- `1` (core)
- `2` (init capture)
- `3` (tracking)

## 6.4 Start controller manually
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 run ibvs_control ibvs_twist_controller'
```

Enable motion:
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 param set /ibvs_twist_controller_node enable_motion true'
```

Disable motion:
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 param set /ibvs_twist_controller_node enable_motion false'
```

## 6.5 Ensure UR twist controller is active
```bash
docker exec -it ros2_ur_driver bash -lc 'source /home/ros_ws/install/setup.bash && ros2 control list_controllers'
```

Activate twist controller if needed:
```bash
docker exec -it ros2_ur_driver bash -lc 'source /home/ros_ws/install/setup.bash && ros2 control switch_controllers --activate cartesian_twist_passthrough_controller --deactivate joint_trajectory_controller scaled_joint_trajectory_controller forward_position_controller forward_velocity_controller passthrough_trajectory_controller'
```

## 6.6 UR run-state checks (critical)
```bash
docker exec -it ros2_ur_driver bash -lc 'source /home/ros_ws/install/setup.bash && ros2 topic echo /io_and_status_controller/robot_program_running --once'
docker exec -it ros2_ur_driver bash -lc 'source /home/ros_ws/install/setup.bash && ros2 topic echo /speed_scaling_state_broadcaster/speed_scaling --once'
```

Required for movement:
- `robot_program_running: true`
- `speed_scaling > 0`

## 7) Manual Node Commands (without manager)
Keypoint:
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 run ibvs_perception keypoint --ros-args -p detector_type:=xfeat -p device:=cpu -p top_k:=1024 -p debug_mode:=true'
```

Reference manager:
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 run ibvs_reference reference_manager --ros-args -p debug_mode:=true'
```

Trigger capture:
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 service call /ibvs/reference/start_capture std_srvs/srv/Trigger "{}"'
```

Matcher:
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 run ibvs_matching descriptor_matcher --ros-args -p match_threshold:=0.85 -p mutual_check:=true'
```

Match viz:
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 run ibvs_matching matches_viz'
```

RViz2:
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && rviz2'
```

## 8) Recommended Test Procedure (Stepwise)
1. Start all containers.
2. Build/source workspace in `ros_ws` container.
3. Start session manager.
4. Manager: `6 -> 1 -> 2 -> 3`.
5. Verify UR state (`robot_program_running=true`, `speed_scaling>0`).
6. Start `ibvs_twist_controller` with `enable_motion=false`.
7. Verify `cartesian_twist_passthrough_controller` is active.
8. Set `enable_motion=true` and perform small pose perturbations.
9. Observe:
   - `/cartesian_twist_passthrough_controller/cmd_vel`
   - controller logs (`rms_px`, `matches`, `Goal reached`)

## 9) Troubleshooting

### Problem: Manager init timeout, but capture seemed to complete
- Cause: init topic polling may fail if QoS/CLI timing is off.
- Current manager is patched to robustly read `/ibvs/init_done` with transient/reliable QoS and longer timeout.

### Problem: Robot does not move although twist is published
Check in order:
1. `cartesian_twist_passthrough_controller` is `active`.
2. `/io_and_status_controller/robot_program_running` is `true`.
3. `/speed_scaling_state_broadcaster/speed_scaling` > `0.0`.
4. External Control program on pendant is running.

### Problem: Robot moves in wrong direction
- Tune axis signs:
  - `axis_sign_vx`, `axis_sign_vy`, `axis_sign_vz`, `axis_sign_wz`
- Calibrate one DOF at a time (`allow_v*` / `allow_w*`).

### Problem: Oscillation near goal
- Lower `lambda_gain`, increase `dls_damping`.
- Reduce speed limits.
- Increase `error_stop_px` and/or `stop_hold_sec`.

### Problem: Match collapse / timeout mid-motion
- Reduce commanded speed.
- Increase feature robustness (detector settings, ROI settings).
- Raise `min_matches` for safety stop behavior.

## 10) Notes for New AI Chat / New Team Member
When opening a new chat/session, include:
- Active container model (`camera_driver`, `ros2_ur_driver`, `ros_ws`).
- Current verified topic names:
  - camera: `/camera/camera/color/image_raw`
  - depth: `/camera/camera/aligned_depth_to_color/image_raw`
- Current IBVS controller defaults from `ibvs_twist_controller_node.py`.
- Whether `cartesian_twist_passthrough_controller` is active.
- Whether `robot_program_running=true` and `speed_scaling>0`.
- Whether session manager or manual node startup is used.

This minimizes re-debugging and avoids repeating controller activation/QoS issues.
