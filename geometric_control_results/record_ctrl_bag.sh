#!/usr/bin/env bash

traj="$1"
if [ -z "$traj" ]; then
  echo "Usage: $(basename "$0") <trajectory_name>"
  exit 1
fi

BAG_ROOT=~/Source/Zouhair/bag_files
ts=$(date +%Y%m%d_%H%M%S)
out="${BAG_ROOT}/${traj}/${traj}_${ts}"

mkdir -p "$(dirname "$out")"

echo "Recording trajectory '${traj}' to:"
echo "  $out"

ros2 bag record -o "$out" \
  /uav_0/control/internal/alpha_des \
  /uav_0/control/internal/e_R \
  /uav_0/control/internal/e_omega \
  /uav_0/control/internal/e_v \
  /uav_0/control/internal/e_x \
  /uav_0/control/internal/omega_des \
  /uav_0/control/internal/pitch_des \
  /uav_0/control/internal/roll_des \
  /uav_0/control/output/M \
  /uav_0/control/output/f \
  /uav_0/control/output/forces \
  /uav_0/control/output/normalized_throttle \
  /uav_0/control/rotor0/ref \
  /uav_0/control/rotor1/ref \
  /uav_0/control/rotor2/ref \
  /uav_0/control/rotor3/ref \
  /uav_0/state/accel \
  /uav_0/state/jerk \
  /uav_0/state/pose \
  /uav_0/state/twist \
  /uav_0/state/twist_inertial \
  /uav_0/trajectory/desired/acceleration \
  /uav_0/trajectory/desired/jerk \
  /uav_0/trajectory/desired/position \
  /uav_0/trajectory/desired/snap \
  /uav_0/trajectory/desired/velocity \
  /uav_0/trajectory/desired/yaw \
  /uav_0/trajectory/desired/yaw_acceleration \
  /uav_0/trajectory/desired/yaw_rate
