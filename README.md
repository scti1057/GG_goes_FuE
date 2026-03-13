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
│  │  ├─ ibvs_filter/
│  │  └─ ibvs_control/
│  └─ third_party/
└─ README.md
```

## 3) High-Level Architecture
Pipeline:
1. `ibvs_perception/keypoint` detects keypoints + descriptors on camera images.
2. `ibvs_reference/reference_manager` captures a robust reference set (time window + consistency counts).
3. `ibvs_matching/descriptor_matcher` matches live descriptors to reference descriptors.
4. `ibvs_filter/filter_node` tracks a bounded active keypoint set with EKF/UKF/ESKF/SKF.
5. `ibvs_filter/filter_debug_node` overlays raw vs. filtered features and filter diagnostics.
6. `ibvs_control/ibvs_twist_controller` computes IBVS twist from selected feature source (`filtered` or `raw`) and publishes to UR twist controller.

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
                                                       +--> filter_node --> /ibvs/filtered_features
                                                       |                  /ibvs/filter/status
                                                       |                  /ibvs/filter/uncertainty
                                                       |
                                                       +--> filter_debug_node --> /ibvs/filter_debug_image
                                                       |
                                                       +--> ibvs_twist_controller (raw/filtered select)
                                                             --> /cartesian_twist_passthrough_controller/cmd_vel
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

## 5.6 `ibvs_filter` - `filter_node`
Node: `ibvs_filter.filter_node`

Purpose:
- Keep a bounded active set of reference-indexed keypoints (`max_active_keypoints`).
- Predict and update active features in image space (EKF/UKF/ESKF/SKF).
- Publish directly the active filtered points (not proxy corners, no proxy model).

Subscriptions:
- `/camera/camera/color/camera_info`
- `/ibvs/reference/keypoints` (transient_local)
- `/ibvs/matches` (raw descriptor matches)
- camera velocity topic (default `/cartesian_twist_passthrough_controller/cmd_vel`)

Publications:
- `/ibvs/filtered_features` (`ibvs_msgs/Matches`, `ref_id` = active keypoint ids, `xy` = filtered state)
- `/ibvs/filter/status` (`std_msgs/String`)
- `/ibvs/filter/uncertainty` (`std_msgs/Float32`, `trace(P)`)
- `/ibvs/filter/update_status` (`std_msgs/String`)
- `/ibvs/filter/update_count` (`std_msgs/UInt32`)
- `/ibvs/filter/update_success_count` (`std_msgs/UInt32`)
- `/ibvs/filter/active_count` (`std_msgs/UInt32`)

Important params:
- `filter_type`: `ekf|ukf|eskf|skf`
- `q_noise`, `r_noise`, `gate_threshold`, `z_depth`, `predict_rate`
- `max_active_keypoints`, `min_init_keypoints`, `min_update_keypoints`
- `camera_velocity_deadband_linear`, `camera_velocity_deadband_angular`, `camera_velocity_stale_timeout`

### 5.6.1 Filter-Ablauf im Detail (pro Zyklus)

Schritt 1: Eingang der Roh-Matches (`/ibvs/matches`, Update-Pfad)
- Für jedes Match wird `ref_id` gegen die Referenzliste geprüft.
- Es werden `current_pixels`, `desired_pixels`, `ref_ids`, `sim` aufgebaut.
- Der Update-Schritt läuft eventbasiert nur, wenn Matches ankommen.

Schritt 2: Aufbereitung und aktives Set
- Doppelte `ref_id` werden entfernt (behalten wird der Eintrag mit höherer `sim`).
- Falls noch kein aktives Set existiert:
  - Es braucht mindestens `min_init_keypoints`.
  - Danach wird ein aktives Set bis `max_active_keypoints` gewählt.
  - Auswahlprinzip: räumliche Verteilung (farthest-point-artig) mit leichtem Score-Bias.
- Aktives Set bedeutet: genau diese `ref_id` liegen im Filterzustand.

Schritt 3: Update gegen beobachtete aktive Punkte
- Aus den aktuellen Matches werden nur die Punkte genutzt, deren `ref_id` im aktiven Set ist.
- Daraus entsteht eine partielle Messung `z` und eine Auswahlmatrix `H_obs` (sparse).
- Gating erfolgt über (normalisierte) Mahalanobis-Distanz.
- Bei erfolgreichem Update: Status `UPDATE` (oder `INIT`/`RELOCALIZED`).
- Bei Fehlschlag: z. B. `REJECT (OUTLIER)`, `REJECT (SINGULAR)`, `REJECT (GEOMETRY)`.

Schritt 4: Predict im Timer-Pfad
- Unabhängig von Matches läuft mit `predict_rate` die Prädiktion.
- Eingang ist der Kameratwist aus dem Velocity-Topic.
- Ergebnisstatus im Timer ist typischerweise `PREDICT` (oder `PREDICT (GEOMETRY HOLD)`).

Schritt 5: Publikation
- `/ibvs/filtered_features` enthält direkt die aktiv gefilterten Zustandspunkte:
  - `ref_id = active_ref_ids`
  - `xy = gefilterte Punkte des aktiven Zustands`
- Zusätzlich werden Meta-Topics publiziert:
  - `/ibvs/filter/status`
  - `/ibvs/filter/uncertainty` (`trace(P)`)
  - `/ibvs/filter/update_status`
  - `/ibvs/filter/update_count`
  - `/ibvs/filter/update_success_count`
  - `/ibvs/filter/active_count`

### 5.6.2 Fallunterscheidung (inkl. "10 Keypoints"-Beispiel)

Fall A: Erstinitialisierung, zu wenige Matches
- Bedingung: noch kein aktives Set und `< min_init_keypoints`.
- Ergebnis: keine Initialisierung, kein Zustandsupdate.
- Status im Update-Pfad bleibt reject/missing.

Fall B: Erstinitialisierung, genug Matches
- Bedingung: noch kein aktives Set und `>= min_init_keypoints`.
- Ergebnis: aktives Set wird aufgebaut, Zustand wird initial aus den gewählten Punkten gesetzt.
- Status: `INIT`.

Fall C: Laufender Betrieb, genug aktive Beobachtungen
- Bedingung: aktives Set vorhanden und beobachtete aktive Punkte `>= min_update_keypoints`.
- Ergebnis: partielles Update nur auf diese beobachteten aktiven IDs.
- Nicht beobachtete aktive Punkte bleiben über Predict im Zustand erhalten.

Fall D: Laufender Betrieb, zu wenige aktive Beobachtungen
- Bedingung: beobachtete aktive Punkte `< min_update_keypoints`.
- Ergebnis: Relokalisierungsversuch aus den aktuellen Roh-Matches.
- Wenn Relokalisierung genug Punkte hat (`>= min_init_keypoints`):
  - neues aktives Set wird gewählt, Zustand neu gesetzt.
  - Status: `RELOCALIZED`.
- Wenn Relokalisierung zu wenige Punkte hat:
  - kein Update; altes aktives Set bleibt bestehen.
  - es läuft nur Predict weiter.

Fall E: Dein Beispiel "10 ursprünglich getrackte Keypoints"
- Angenommen `N_active = 10`, `min_update_keypoints = 4`.
- Wenn in einem Frame 6 dieser 10 wiedergefunden werden:
  - Update läuft auf diesen 6.
  - Die fehlenden 4 bleiben im Zustand und werden nur prädiziert.
- Wenn nur 3 von 10 wiedergefunden werden:
  - Update ist zu schwach (`3 < 4`), Relokalisierung wird versucht.
  - Dann können auch bisher nicht aktive Matches ins neue aktive Set aufgenommen werden.
  - Die "ursprünglichen 10" sind nicht fest reserviert; aktives Set kann ersetzt werden.

Fall F: Was passiert mit "den restlichen" (nicht aktiven) Matches?
- Im normalen Update: sie werden ignoriert (sie sind nicht Teil des Zustands).
- Bei Initialisierung/Relokalisierung: sie sind Kandidaten und können aktiv werden.

### 5.6.3 Auswahl des aktiven Sets (wann es sich ändert)

Grundidee:
- Das aktive Set ist eine feste Liste von `ref_id` im aktuellen Filterzustand.
- Diese Liste wird **nicht** bei jedem normalen Mess-Update neu gewählt.
- Der Set-Wechsel ist **messungsgetrieben** und passiert nur im Update-Pfad (`/ibvs/matches`), nicht im Predict-Timer.
- Normale Messungen aktualisieren nur die Zustandswerte der bereits aktiven IDs.

Wie wird das Set gewählt?
- Kandidaten sind die aktuell gültigen Roh-Matches (`ref_id`, `xy`, `sim`).
- Doppelte `ref_id` werden zuerst aufgelöst (behalten wird der Match mit höherer `sim`).
- Danach Auswahl bis `max_active_keypoints` mit:
  - starker räumlicher Abdeckung (farthest-point-artige Verteilung),
  - leichtem Score-Bias aus `sim`.
- Set-Größe nach Auswahl: `N_active = min(max_active_keypoints, Anzahl eindeutiger Kandidaten)`.

Wann ändert sich das aktive Set über die Messung?
- Bei **Erstinitialisierung**:
  - Bedingung: kein aktives Set + mindestens `min_init_keypoints`.
- Bei **Relokalisierung**:
  - Bedingung: beobachtete aktive Punkte `< min_update_keypoints`.
  - Dann wird aus den aktuellen Roh-Matches ein neues aktives Set gewählt
    (wenn diese mindestens `min_init_keypoints` liefern).
- Bei **externer Relokalisierung**:
  - z. B. neue Referenz (`force_relocalization()`), danach wird beim nächsten
    ausreichenden Match-Frame neu gewählt.

Explizite Reihenfolge pro eingehender `Matches`-Messung:
- 1) Wenn es noch kein aktives Set gibt: Initialisierungsversuch aus dieser Messung.
- 2) Wenn es ein aktives Set gibt: nur Matches mit aktiver `ref_id` zählen als beobachtet.
- 3) Falls beobachtete aktive Punkte `>= min_update_keypoints`: normales partielles Update, Set bleibt gleich.
- 4) Falls beobachtete aktive Punkte `< min_update_keypoints`: Relokalisierungsversuch aus allen aktuellen Roh-Matches.
- 5) Relokalisierung nur erfolgreich bei `>= min_init_keypoints` (nach Deduplizierung), sonst kein Set-Wechsel.

Wann ändert es sich **nicht**?
- Wenn genügend aktive Punkte beobachtet werden (`>= min_update_keypoints`):
  - dann bleibt die aktive ID-Liste gleich,
  - nur Zustandswerte/Kovarianz werden geupdatet.
- Auch wenn viele gute nicht aktive Matches verfügbar sind, werden diese in diesem Fall
  nicht „on-the-fly“ ins Set gemischt.
- Im reinen Predict-Betrieb (Timer ohne neue Matches) gibt es nie einen Set-Wechsel.

Einfluss von Parameteränderungen zur Laufzeit:
- Änderungen an `max_active_keypoints`, `min_init_keypoints`, `min_update_keypoints`
  wirken sofort auf die Regeln.
- Das bestehende aktive Set wird aber typischerweise erst bei der nächsten
  Initialisierung/Relokalisierung neu zusammengesetzt.

### 5.6.4 Unsicherheit `P` (wie sie geführt wird)

Grundprinzip:
- `P` ist die Kovarianzmatrix des Filterzustands.
- Publiziert wird `trace(P)` auf `/ibvs/filter/uncertainty`.

Initialisierung:
- Start unsicher (`P` typischerweise groß, diagonal skaliert, z. B. `1000 * I`).
- Bei `force_relocalization()` wird `initialized=false`; je nach Filtertyp wird `P` hochgesetzt.

Prädiktion:
- EKF: `P = F P F^T + Q` mit numerischer Jacobimatrix `F`.
- UKF: `P` aus Sigma-Punkten rekonstruiert, dann `+ Q`.
- ESKF: Fehlerkovarianz `P = F_dx P F_dx^T + Q`.
- SKF: konstantes Geschwindigkeitsmodell mit Zustandsdimension `4N` (Position+Geschwindigkeit).

Update:
- Es wird nur auf beobachtete aktive Punkte aktualisiert (`H_obs` ist Auswahlmatrix).
- Messrauschen `R` wirkt nur auf die beobachteten Messkomponenten.
- Kovarianz wird klassisch mit Kalman-Gain reduziert (Form `P <- (I-KH)P`).

Wichtige Beobachtung:
- Wenn über längere Zeit nur Predict läuft (wenig/keine Updates), steigt Unsicherheit typischerweise.
- Bei regelmäßigen erfolgreichen Updates sinkt `trace(P)` wieder.

## 5.7 `ibvs_filter` - `filter_debug_node`
Node: `ibvs_filter.filter_debug_node`

Purpose:
- Separate debug visualization process (decoupled from filter runtime).
- Overlay reference/raw/filtered points and filter diagnostics.

Subscriptions:
- camera image topic (default `/camera/camera/color/image_raw/compressed`)
- `/ibvs/reference/keypoints`
- `/ibvs/matches` (raw)
- `/ibvs/filtered_features` (filtered)
- `/ibvs/filter/status`
- `/ibvs/filter/uncertainty`
- `/ibvs/filter/update_status`
- `/ibvs/filter/update_count`
- `/ibvs/filter/update_success_count`
- `/ibvs/filter/active_count`

Publications:
- `/ibvs/filter_debug_image`

## 5.8 `ibvs_control` - `ibvs_twist_controller`
Node: `ibvs_control.ibvs_twist_controller_node`

Purpose:
- Compute IBVS Cartesian twist from `reference + selected feature source`.
- Enforce safety gates and stop criteria.
- Publish command twist for UR passthrough controller.

Subscriptions:
- `/ibvs/matches` (raw)
- `/ibvs/filtered_features` (filtered)
- `/ibvs/reference/keypoints` (transient_local)
- `/ibvs/init_done` (transient_local)

Publications:
- `/cartesian_twist_passthrough_controller/cmd_vel`
- `/ibvs/control/goal_reached` (transient_local)

Current default control params (tuned):
- `lambda_gain=0.36`
- `dls_damping=0.1`
- `feature_source=filtered`
- `filtered_fallback_to_raw=true`
- `min_matches=60`
- `error_stop_px=10.0`
- `stop_hold_sec=0.8`
- `max_linear_speed=0.012`
- `max_angular_speed=0.15`
- DOFs: `vx,vy,vz,wz=true`, `wx,wy=false`
- axis signs: `vx=-1`, `vy=+1`, `vz=0`, `wz=-1`
- `enable_motion=false` by default (safety)

Notes:
- If camera mount or frame convention changes, axis signs likely need retuning.
- Keep `enable_motion=false` until system state and controllers are confirmed.

## 5.9 Session Tool - `ibvs_session_manager.py`
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
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && python3 /home/ros_ws/scripts/ibvs_session_manager.py --debug --device=cuda'
```

Typical flow in manager:
- `6` (clean stop)
- `1` (core)
- `2` (init capture)
- `3` (tracking)

Note:
- `ibvs_filter/filter_node` and `ibvs_filter/filter_debug_node` are started separately (not by the manager menu).

## 6.4 Start controller manually
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 run ibvs_control ibvs_twist_controller'
```

Set feature source to filtered (default):
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 param set /ibvs_twist_controller_node feature_source filtered'
```

Set feature source to raw:
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 param set /ibvs_twist_controller_node feature_source raw'
```

Set min number of points:
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 param set /ibvs_twist_controller_node min_matches 20'
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

Filter:
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 run ibvs_filter filter_node'
```

Filter debug:
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 run ibvs_filter filter_debug_node'
```

RViz2:
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && rviz2'
```

Constant Velocity (drive fo 10s in positive x-Axis with 1cm/s=0.01m/s)
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && timeout 10s ros2 topic pub -r 50 /cartesian_twist_passthrough_controller/cmd_vel geometry_msgs/msg/Twist \ "{linear: {x: 0.01, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"'
```

## 8) Recommended Test Procedure (Stepwise)
1. Start all containers.
2. Build/source workspace in `ros_ws` container.
3. Start session manager.
4. Manager: `6 -> 1 -> 2 -> 3`.
5. Verify UR state (`robot_program_running=true`, `speed_scaling>0`).
6. Start `ibvs_twist_controller` with `enable_motion=false`.
7. Optionally start `filter_node` and `filter_debug_node` if filtered IBVS should be used.
8. Verify `cartesian_twist_passthrough_controller` is active.
9. Set `enable_motion=true` and perform small pose perturbations.
10. Observe:
   - `/cartesian_twist_passthrough_controller/cmd_vel`
   - `/ibvs/filtered_features`, `/ibvs/filter/active_count`
   - controller logs (`rms_px`, `points`, `Goal reached`)

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
- If using `feature_source=filtered`, decide whether `filtered_fallback_to_raw` should be enabled.

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

## 11) Filter Tuning (ibvs_filter)

Start filter node:

EKF example (`q=2.0, r=1.1, gate=20.0, z=0.25`, active set up to 12):
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 run ibvs_filter filter_node --ros-args -p filter_type:=ekf -p q_noise:=2.0 -p r_noise:=1.1 -p gate_threshold:=20.0 -p z_depth:=0.25 -p max_active_keypoints:=12 -p min_init_keypoints:=8 -p min_update_keypoints:=1'
```

Start debug overlay node:
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 run ibvs_filter filter_debug_node'
```

Live tuning (without restart):
```bash
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 param set /ibvs_filter_node q_noise 0.5'
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 param set /ibvs_filter_node r_noise 80.0'
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 param set /ibvs_filter_node gate_threshold 20.0'
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 param set /ibvs_filter_node z_depth 0.45'
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 param set /ibvs_filter_node force_relocalization true'
docker exec -it ros_ws bash -lc 'source /home/ros_ws/install/setup.bash && ros2 param set /ibvs_filter_node min_update_keypoints 1'
```

Hinweis:
- `force_relocalization=true` triggert eine manuelle Relokalisierung sofort.
- Falls dein Param-Client identische Werte nicht erneut schreibt: für einen weiteren Trigger kurz auf `false` und danach wieder auf `true` setzen.

Practical interpretation:
- `q_noise` up: model less trusted, filter follows measurements more quickly.
- `q_noise` down: smoother prediction, but can lag on fast motion.
- `r_noise` up: measurements less trusted, stronger smoothing.
- `r_noise` down: more reactive to matches, but noisier.
- `gate_threshold` up: fewer outlier rejects; down: stricter reject behavior.
- `max_active_keypoints` up: more geometric coverage, but higher compute cost.
- `min_init_keypoints` / `min_update_keypoints` too high: relocalization can happen too often.

Recommended tuning sequence:
1. Keep robot/camera static. Increase `r_noise` until jitter of yellow filtered points is visibly reduced.
2. Move slowly in one axis. Increase `q_noise` until lag is acceptable without noisy oscillation.
3. Introduce occasional bad matches (partial occlusion). Decrease `gate_threshold` until outliers are rejected, then back off slightly.
4. Recheck with your normal motion speed.

Useful gate references:
- The effective Mahalanobis dimension is `2 * (#observed active points)` and changes frame-to-frame.
- Start around `20.0` and tune empirically for your scene/outlier rate.
