#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${ROS_WS_CONTAINER:-ros_ws}"
MATCHER_NODE="${MATCHER_NODE:-/descriptor_matcher_node}"
DURATION_SEC="${1:-60}"
WARMUP_SEC="${WARMUP_SEC:-3}"
RUN_NAME="${2:-local_rescue_bench_$(date +%Y%m%d_%H%M%S)}"
MODE="${3:-all}"  # all | off | shadow | active
OUT_ROOT="/home/ros_ws/logs/local_rescue_benchmark/${RUN_NAME}"

TOPICS=(
  /ibvs/matching/local_rescue_stats
  /ibvs/matching/local_rescue_attempts
  /ibvs/matching/local_rescue_success
  /ibvs/matching/local_rescue_reject
  /ibvs/filter/active_count
  /ibvs/filter/update_success_count
  /ibvs/filtered_features
  /ibvs/matches
  /ibvs/filter/uncertainty
)

if ! [[ "${DURATION_SEC}" =~ ^[0-9]+$ ]] || [ "${DURATION_SEC}" -le 0 ]; then
  echo "DURATION_SEC must be a positive integer, got: ${DURATION_SEC}" >&2
  exit 1
fi

if ! [[ "${MODE}" =~ ^(all|off|shadow|active)$ ]]; then
  echo "MODE must be one of: all|off|shadow|active, got: ${MODE}" >&2
  echo "Usage: $0 <duration_sec> [run_name] [mode]" >&2
  exit 1
fi

if [ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" != "true" ]; then
  echo "Container '${CONTAINER_NAME}' is not running." >&2
  exit 1
fi

dexec() {
  docker exec -i "${CONTAINER_NAME}" bash -lc "$1"
}

join_by_space() {
  local IFS=' '
  echo "$*"
}

TOPIC_ARGS="$(join_by_space "${TOPICS[@]}")"
SETUP='source /home/ros_ws/install/setup.bash'

# Basic readiness checks.
if ! dexec "${SETUP} && ros2 node list | grep -Fx '${MATCHER_NODE}'" >/dev/null 2>&1; then
  echo "Matcher node ${MATCHER_NODE} not found. Start your stack first." >&2
  exit 1
fi

if ! dexec "${SETUP} && ros2 topic list | grep -Fx '/ibvs/matches'" >/dev/null 2>&1; then
  echo "Topic /ibvs/matches not found. Verify pipeline is running." >&2
  exit 1
fi

echo "Creating output root: ${OUT_ROOT}"
dexec "mkdir -p '${OUT_ROOT}'"

echo "Benchmark run: ${RUN_NAME}"
echo "Duration per mode: ${DURATION_SEC}s (warmup ${WARMUP_SEC}s)"
if [ "${MODE}" = "all" ]; then
  echo "Modes: off -> shadow -> active"
else
  echo "Mode: ${MODE}"
fi

do_mode() {
  local mode="$1"
  local mode_dir="${OUT_ROOT}/${mode}"

  echo
  echo "=== Mode: ${mode} ==="
  dexec "${SETUP} && ros2 param set ${MATCHER_NODE} local_rescue_mode '\"${mode}\"'" >/dev/null
  echo "Set ${MATCHER_NODE}.local_rescue_mode=${mode}"

  # Snapshot key matcher params for reproducibility.
  dexec "${SETUP} && {
    echo \"mode=${mode}\";
    ros2 param get ${MATCHER_NODE} sim_floor;
    ros2 param get ${MATCHER_NODE} use_adaptive_gates;
    ros2 param get ${MATCHER_NODE} adaptive_radius_min_px;
    ros2 param get ${MATCHER_NODE} adaptive_radius_max_px;
    ros2 param get ${MATCHER_NODE} adaptive_sim_threshold_min;
    ros2 param get ${MATCHER_NODE} adaptive_sim_threshold_max;
    ros2 param get ${MATCHER_NODE} kp_sigma_low_px;
    ros2 param get ${MATCHER_NODE} kp_sigma_high_px;
    ros2 param get ${MATCHER_NODE} local_ambiguity_min_score_gap;
    ros2 param get ${MATCHER_NODE} local_ambiguity_min_score_ratio;
  } > '${mode_dir}_params.txt'"

  [ "${WARMUP_SEC}" -gt 0 ] && sleep "${WARMUP_SEC}"

  echo "Recording rosbag to: ${mode_dir}"
  set +e
  dexec "${SETUP} && timeout ${DURATION_SEC}s ros2 bag record -o '${mode_dir}' ${TOPIC_ARGS}"
  local rc=$?
  set -e

  if [ "${rc}" -ne 0 ] && [ "${rc}" -ne 124 ]; then
    echo "ros2 bag record failed for mode ${mode} with exit code ${rc}" >&2
    exit "${rc}"
  fi

  echo "Finished mode ${mode}."
}

if [ "${MODE}" = "all" ]; then
  do_mode off
  do_mode shadow
  do_mode active
else
  do_mode "${MODE}"
fi

# Safety: leave matcher in baseline mode.
dexec "${SETUP} && ros2 param set ${MATCHER_NODE} local_rescue_mode '\"off\"'" >/dev/null || true

echo
echo "Benchmark complete. Bags saved under: ${OUT_ROOT}"
echo "Quick check:"
echo "  docker exec -it ${CONTAINER_NAME} bash -lc 'ls -lah ${OUT_ROOT}'"
