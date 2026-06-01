# Geometric Tracking Control of a Quadrotor UAV on SE(3)

**MEng Project — University of Toronto (UTIAS) | FSC Lab | 2025–2026**

Implementation and simulation of the geometric tracking controller from Lee, Leok, and McClamroch ("Geometric Tracking Control of a Quadrotor UAV on SE(3)", CDC 2010) applied to a custom DJI F450 quadrotor model in NVIDIA IsaacSim via ROS2.

---

## Overview

Geometric control operates directly on the SE(3) manifold, avoiding the singularities and ambiguities of Euler-angle representations. This project:

1. Studies the theory and stability proofs of the SE(3) geometric controller
2. Implements and validates a standalone Python proof-of-concept simulator
3. Models the F450 quadrotor in OnShape CAD and exports it as a USD asset for IsaacSim
4. Extends [fsc_PegasusSimulator](https://github.com/adamakio/fsc_PegasusSimulator) to spawn and control the custom F450 via ROS2
5. Packages the controller and trajectory generation as ROS2 nodes and tests them on hover, circle, figure-eight, and elliptic helix trajectories

---

## Repository Structure

```
geometric_control_poc/          # Standalone Python PoC — validated before IsaacSim
├── simulate_geometric_control.py
└── sim_figures/                # Hover, step, circle, helix, flip trajectories

fsc_geometric_controller/       # ROS2 package — Geometric Tracking Controller
├── fsc_geometric_controller/
│   ├── geometric_control.py    # SE(3) controller: hat/vee maps, SO(3) error,
│   │                           #   heading projection, thrust + moment computation
│   └── geometric_control_node.py
├── config/params_geometric_control.yaml
└── launch/geometric_control_baseline_launch.py

fsc_geometric_control_trajectories/  # ROS2 package — trajectory generation
├── fsc_geometric_control_trajectories/
│   ├── trajectories.py         # Hover, circle, figure-eight, elliptic helix
│   └── trajectory_publisher_node.py
├── config/trajectory.yaml
└── launch/trajectory.launch.py

fsc_quadrotor_sim/              # ROS2 package — standalone quadrotor dynamics node
├── fsc_quadrotor_sim/
│   ├── quadrotor_dynamics.py   # Rigid-body dynamics, RK4 integration, F450 params
│   └── quadrotor_sim_node.py
├── config/params_quadrotor_sim.yaml
└── launch/quadrotor_sim_launch.py

geometric_control_results/      # Simulation results
├── report_plots/               # Position, attitude, error, and control plots
│   ├── hover/
│   ├── circle/
│   ├── figure_eight/
│   └── elliptic_helix/
├── plot_controller_report.py   # Plot generation from ROS2 bag files
└── record_ctrl_bag.sh          # Bag recording script

geometric_control_figures/      # Diagrams used in the report
figures/                        # CAD renders and technical drawings
Videos/                         # IsaacSim screen recordings

Final Report/
├── MEng_FinalReport_ZouhairHamaimou_1004891986.pdf
└── report.tex
```

---

## Geometric Controller

The controller is implemented in [fsc_geometric_controller/geometric_control.py](fsc_geometric_controller/fsc_geometric_controller/geometric_control.py) and follows the Lee et al. formulation with a body-frame convention where b3 points up:

**Position error and feedforward:**
```
A = -kx·ex - kv·ev + mg·e3 + m·a_des
f = dot(A, R·e3)
```

**Desired rotation (heading projection):**
```
b3_des = A / ||A||
b1_des = b1_c - dot(b1_c, b3_des)·b3_des  (projected heading)
R_des  = [b2_des×b3_des, b2_des, b3_des]
```

**Attitude error (SO(3)):**
```
eR = 0.5 · vee(R_des^T · R - R^T · R_des)
eΩ = Ω - R^T · R_des · Ω_des
M  = -kR·eR - kΩ·eΩ + Ω×J·Ω - J·(Ω̂·R^T·R_des·Ω_des - R^T·R_des·Ω̇_des)
```

---

## F450 CAD and USD Model

The DJI F450 was modelled from scratch in OnShape with accurate physical properties (mass, inertia tensor, rotor placement, thrust curves) derived from manufacturer specs and thrust stand measurements.

The model was exported as a USD file via the NVIDIA OnShape importer and integrated into [fsc_PegasusSimulator](https://github.com/adamakio/fsc_PegasusSimulator) (see `extensions/pegasus.simulator/pegasus/simulator/assets/Robots/F450/`).

---

## Proof-of-Concept Simulation Results

The standalone Python PoC ([geometric_control_poc/simulate_geometric_control.py](geometric_control_poc/simulate_geometric_control.py)) validates the controller across five maneuvers before IsaacSim deployment:

| Test | Description |
|------|-------------|
| Hover | Stabilize from near-zero initial conditions |
| Step | 1 m position step in x |
| Circle | 1 m radius horizontal circle |
| Helix | Ascending spiral |
| Flip | 360° roll (attitude only) |

---

## IsaacSim Results

Closed-loop results with the F450 USD model in IsaacSim, controlled via ROS2:

| Trajectory | Position RMSE |
|------------|:-------------:|
| Hover | see report |
| Circle | see report |
| Figure-eight | see report |
| Elliptic helix | see report |

Result plots are in [geometric_control_results/report_plots/](geometric_control_results/report_plots/).

---

## Dependencies

**ROS2 packages:** ROS2 Humble or later

**Python:** NumPy · SciPy (PoC only)

**Simulator:** NVIDIA IsaacSim 4.x · [fsc_PegasusSimulator](https://github.com/adamakio/fsc_PegasusSimulator) (f450_dev branch)

---

## Report

[MEng_FinalReport_ZouhairHamaimou_1004891986.pdf](Final%20Report/MEng_FinalReport_ZouhairHamaimou_1004891986.pdf)
