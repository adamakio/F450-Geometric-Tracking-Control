#!/usr/bin/env python3
"""
Generate E1-E5 comparison plots for the elliptic-helix experiments.

Expected working directory:
    .
    ├── ideal_quadrotor
    ├── f450_wrench
    ├── f450_throttle
    ├── f450_sim_rk4
    ├── f450_sim_euler
    └── plot_experiment_comparison.py

Outputs:
    experiment_plots/*.png
    experiment_plots/metrics.txt
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plot_controller_report import build_dataset, load_bag_topics, safe_norm_rows


ROOT = Path.cwd()
FIGURES_DIR = ROOT / "experiment_plots"
METRICS_PATH = FIGURES_DIR / "metrics.txt"
MAX_METRIC_TRANSIENT_S = 5.0
METRICS_ONLY_FLAG = "--metrics-only"

EXPERIMENTS: Dict[str, Dict[str, str]] = {
    "E1": {"bag": "ideal_quadrotor", "out": "E1_ideal_block", "label": r"$E_1$"},
    "E2": {"bag": "f450_wrench", "out": "E2_f450_lumped", "label": r"$E_2$"},
    "E3": {"bag": "f450_throttle", "out": "E3_f450_distributed", "label": r"$E_3$"},
    "E4": {"bag": "f450_sim_rk4", "out": "E4_custom_rk4", "label": r"$E_4$"},
    "E5": {"bag": "f450_sim_euler", "out": "E5_custom_euler", "label": r"$E_5$"},
}

EXPERIMENT_ORDER = ("E1", "E2", "E3", "E4", "E5")
EXPERIMENT_COLORS = {
    "E1": "#1f77b4",
    "E2": "#ff7f0e",
    "E3": "#2ca02c",
    "E4": "#d62728",
    "E5": "#9467bd",
}
EXPERIMENT_STYLES = {
    "E1": "-",
    "E2": "--",
    "E3": "-.",
    "E4": (0, (5, 1.6)),
    "E5": (0, (1.2, 1.2)),
}
SAVED_OUTPUTS: List[Tuple[Path, str, Tuple[float, float]]] = []


@dataclass
class ExperimentData:
    key: str
    dataset: Dict[str, np.ndarray]
    out_dir: Path
    active_start: float
    active_end: float


def setup_matplotlib() -> None:
    latex_available = shutil.which("latex") is not None
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
        "axes.linewidth": 0.8,
        "axes.formatter.use_mathtext": True,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "text.usetex": latex_available,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#d0d0d0",
    })


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_current_figure(path: Path, description: str, tight: bool = True) -> None:
    fig = plt.gcf()
    size_inches = tuple(float(v) for v in fig.get_size_inches())
    if tight:
        fig.tight_layout()
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    SAVED_OUTPUTS.append((png_path, description, size_inches))


def relative_figure_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_deg_180(angle_deg: np.ndarray) -> np.ndarray:
    return (angle_deg + 180.0) % 360.0 - 180.0


def active_window_from_topics(topics: Dict[str, object]) -> Tuple[float, float]:
    vel_topic = topics["/uav_0/trajectory/desired/velocity"]
    yaw_rate_topic = topics["/uav_0/trajectory/desired/yaw_rate"]

    vel_norm = safe_norm_rows(vel_topic.y)
    yaw_rate = np.abs(yaw_rate_topic.y[:, 0])

    moving_vel = vel_norm > 1e-6
    moving_yaw = yaw_rate > 1e-6

    active_samples = []
    if np.any(moving_vel):
        active_samples.append(vel_topic.t[moving_vel])
    if np.any(moving_yaw):
        active_samples.append(yaw_rate_topic.t[moving_yaw])

    if not active_samples:
        return 0.0, float("inf")

    active_t = np.concatenate(active_samples)
    return float(np.min(active_t)), float(np.max(active_t))


def crop_dataset(ds: Dict[str, np.ndarray], t_start: float, t_end: float) -> Dict[str, np.ndarray]:
    cropped: Dict[str, np.ndarray] = {}
    mask = np.logical_and(ds["t"] >= t_start, ds["t"] <= t_end)
    if not np.any(mask):
        raise RuntimeError("Cropping produced an empty dataset.")

    t0 = float(ds["t"][mask][0])
    for key, value in ds.items():
        if value is None:
            cropped[key] = value
            continue
        if isinstance(value, np.ndarray) and value.shape[0] == ds["t"].shape[0]:
            part = value[mask].copy()
            if key == "t":
                part = part - t0
            cropped[key] = part
        else:
            cropped[key] = value
    return cropped


def trim_to_duration(ds: Dict[str, np.ndarray], duration: float) -> Dict[str, np.ndarray]:
    trimmed: Dict[str, np.ndarray] = {}
    mask = ds["t"] <= duration
    if not np.any(mask):
        raise RuntimeError("Common-duration trim produced an empty dataset.")

    for key, value in ds.items():
        if value is None:
            trimmed[key] = value
            continue
        if isinstance(value, np.ndarray) and value.shape[0] == ds["t"].shape[0]:
            trimmed[key] = value[mask].copy()
        else:
            trimmed[key] = value
    return trimmed


def load_experiments() -> Dict[str, ExperimentData]:
    experiments: Dict[str, ExperimentData] = {}
    common_duration = float("inf")

    for key in EXPERIMENT_ORDER:
        config = EXPERIMENTS[key]
        bag_dir = ROOT / config["bag"]
        out_dir = ensure_dir(FIGURES_DIR / config["out"])
        if not bag_dir.exists():
            raise FileNotFoundError(f"Missing experiment folder: {bag_dir}")

        topics = load_bag_topics(bag_dir)
        active_start, active_end = active_window_from_topics(topics)
        ds = build_dataset(topics)
        ds = crop_dataset(ds, active_start, active_end)
        common_duration = min(common_duration, float(ds["t"][-1]))
        experiments[key] = ExperimentData(
            key=key,
            dataset=ds,
            out_dir=out_dir,
            active_start=active_start,
            active_end=active_end,
        )

    for key in EXPERIMENT_ORDER:
        experiments[key].dataset = trim_to_duration(experiments[key].dataset, common_duration)

    return experiments


def first_reference(experiments: Dict[str, ExperimentData], key: str) -> Tuple[np.ndarray, np.ndarray]:
    for exp_key in EXPERIMENT_ORDER:
        ds = experiments[exp_key].dataset
        return ds["t"], ds[key]
    raise RuntimeError(f"No reference available for {key}")


def normalize_throttles(experiments: Dict[str, ExperimentData]) -> None:
    arrays: List[np.ndarray] = []
    for exp in experiments.values():
        throttle = exp.dataset.get("throttle")
        if throttle is not None and throttle.size:
            arrays.append(np.asarray(throttle, dtype=float))

    if not arrays:
        return

    stacked = np.concatenate([arr.reshape(-1) for arr in arrays])
    global_min = float(np.min(stacked))
    global_max = float(np.max(stacked))
    scale = global_max - global_min
    if scale < 1e-12:
        scale = 1.0

    for exp in experiments.values():
        throttle = exp.dataset.get("throttle")
        if throttle is None:
            continue
        exp.dataset["throttle_raw"] = throttle.copy()
        exp.dataset["throttle"] = np.clip((throttle - global_min) / scale, 0.0, 1.0)


def metrics_only_requested() -> bool:
    return METRICS_ONLY_FLAG in sys.argv[1:]


def configure_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel(r"Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major")
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.4)


def plot_scalar_comparison(
    experiments: Dict[str, ExperimentData],
    out_path: Path,
    description: str,
    ylabel: str,
    actual_getter,
    reference_getter=None,
    legend_ncol: int = 3,
    figsize: Tuple[float, float] = (7.0, 4.0),
    t_min: Optional[float] = None,
) -> None:
    _, ax = plt.subplots(figsize=figsize)

    if reference_getter is not None:
        t_ref, y_ref = reference_getter()
        ref_mask = np.ones_like(t_ref, dtype=bool) if t_min is None else (t_ref >= t_min)
        ax.plot(t_ref[ref_mask], y_ref[ref_mask], color="black", linestyle=":", linewidth=2.4, label=r"Reference")

    for key in EXPERIMENT_ORDER:
        exp = experiments[key]
        t = exp.dataset["t"]
        mask = np.ones_like(t, dtype=bool) if t_min is None else (t >= t_min)
        ax.plot(
            t[mask],
            actual_getter(exp.dataset)[mask],
            color=EXPERIMENT_COLORS[key],
            linestyle=EXPERIMENT_STYLES[key],
            linewidth=2.0,
            label=EXPERIMENTS[key]["label"],
        )

    configure_axes(ax, ylabel)
    ax.legend(ncol=legend_ncol, loc="best")
    save_current_figure(out_path, description)


def plot_3d_trajectory(experiments: Dict[str, ExperimentData], out_path: Path, description: str) -> None:
    fig = plt.figure(figsize=(7.1, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    _, x_ref = first_reference(experiments, "x_des")
    ax.plot(
        x_ref[:, 0], x_ref[:, 1], x_ref[:, 2],
        color="black", linestyle=":", linewidth=2.4, label=r"Reference"
    )

    all_points = [x_ref]
    for key in EXPERIMENT_ORDER:
        x = experiments[key].dataset["x"]
        all_points.append(x)
        ax.plot(
            x[:, 0], x[:, 1], x[:, 2],
            color=EXPERIMENT_COLORS[key],
            linestyle=EXPERIMENT_STYLES[key],
            linewidth=2.0,
            label=EXPERIMENTS[key]["label"],
        )

    ax.set_xlabel(r"$x$ [m]", labelpad=8)
    ax.set_ylabel(r"$y$ [m]", labelpad=8)
    ax.set_zlabel(r"$z$ [m]", labelpad=8)
    ax.grid(True)
    ax.view_init(elev=24, azim=-58)
    ax.legend(ncol=3, loc="upper center")

    pts = np.vstack(all_points)
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.55 * max(np.max(maxs - mins), 1e-3)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 0.85))

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    save_current_figure(out_path, description, tight=False)


def plot_error_components(
    experiments: Dict[str, ExperimentData],
    out_path: Path,
    description: str,
    error_getter,
    ylabel: str,
    component_index: int,
    component_label: str,
) -> None:
    _, ax = plt.subplots(figsize=(7.4, 2.45))

    for key in EXPERIMENT_ORDER:
        err = error_getter(experiments[key].dataset)
        line_label = rf"{EXPERIMENTS[key]['label']}"
        ax.plot(
            experiments[key].dataset["t"],
            err[:, component_index],
            color=EXPERIMENT_COLORS[key],
            linestyle=EXPERIMENT_STYLES[key],
            linewidth=2.0,
            alpha=0.95,
            label=line_label,
        )

    configure_axes(ax, ylabel)
    ax.legend(ncol=2, loc="best", title=component_label)
    save_current_figure(out_path, description)


def plot_tracking_component(
    experiments: Dict[str, ExperimentData],
    out_path: Path,
    description: str,
    actual_key: str,
    desired_key: str,
    component_index: int,
    ylabel: str,
) -> None:
    plot_scalar_comparison(
        experiments,
        out_path,
        description,
        ylabel,
        actual_getter=lambda ds, i=component_index, key=actual_key: ds[key][:, i],
        reference_getter=lambda i=component_index, key=desired_key: (
            lambda t_ref, y_ref: (t_ref, y_ref[:, i])
        )(*first_reference(experiments, key)),
        legend_ncol=3,
        figsize=(7.4, 2.45),
    )


def plot_tracking_component_with_transform(
    experiments: Dict[str, ExperimentData],
    out_path: Path,
    description: str,
    actual_key: str,
    desired_key: str,
    component_index: int,
    ylabel: str,
    actual_transform,
    desired_transform,
) -> None:
    plot_scalar_comparison(
        experiments,
        out_path,
        description,
        ylabel,
        actual_getter=lambda ds, i=component_index, key=actual_key: actual_transform(ds[key][:, i]),
        reference_getter=lambda i=component_index, key=desired_key: (
            lambda t_ref, y_ref: (t_ref, desired_transform(y_ref[:, i]))
        )(*first_reference(experiments, key)),
        legend_ncol=3,
        figsize=(7.4, 2.45),
    )


def plot_throttle_components(experiments: Dict[str, ExperimentData], out_dir: Path) -> None:
    for rotor_idx in range(4):
        plot_scalar_comparison(
            experiments,
            out_dir / f"control_throttle_rotor{rotor_idx + 1}",
            f"Normalized throttle command for rotor {rotor_idx + 1} across E1-E5.",
            r"Normalized throttle",
            actual_getter=lambda ds, i=rotor_idx: ds["throttle"][:, i],
            reference_getter=None,
            legend_ncol=2,
            figsize=(3.55, 2.45),
            t_min=1.0,
        )


def compute_metrics(ds: Dict[str, np.ndarray]) -> Dict[str, float]:
    e_x = ds["e_x"]
    e_v = ds["e_v"]
    e_R = ds["e_R"]
    e_omega = ds["e_omega"]

    pos_norm = safe_norm_rows(e_x)
    vel_norm = safe_norm_rows(e_v)
    att_norm = safe_norm_rows(e_R)
    omega_norm = safe_norm_rows(e_omega)

    post_transient = ds["t"] >= MAX_METRIC_TRANSIENT_S
    if not np.any(post_transient):
        post_transient = np.ones_like(ds["t"], dtype=bool)

    metrics = {
        "RMS_e_x": float(np.sqrt(np.mean(pos_norm[post_transient] ** 2))),
        "max_e_x": float(np.max(pos_norm[post_transient])),
        "RMS_e_v": float(np.sqrt(np.mean(vel_norm[post_transient] ** 2))),
        "RMS_e_R": float(np.sqrt(np.mean(att_norm[post_transient] ** 2))),
        "max_e_R": float(np.max(att_norm[post_transient])),
        "RMS_e_Omega": float(np.sqrt(np.mean(omega_norm[post_transient] ** 2))),
    }

    throttle = ds.get("throttle")
    if throttle is not None:
        sat = np.logical_or(np.isclose(throttle, 0.0), np.isclose(throttle, 1.0))
        metrics["sigma_u"] = float(np.mean(sat))
    else:
        metrics["sigma_u"] = float("nan")

    return metrics


def write_metrics_table(experiments: Dict[str, ExperimentData], out_path: Path) -> None:
    metrics_by_exp = {key: compute_metrics(exp.dataset) for key, exp in experiments.items()}
    rows = [
        ("RMS position error", "RMS_e_x"),
        ("Max position error", "max_e_x"),
        ("RMS velocity error", "RMS_e_v"),
        ("RMS attitude error", "RMS_e_R"),
        ("Max attitude error", "max_e_R"),
        ("RMS angular-velocity error", "RMS_e_Omega"),
        ("Throttle saturation fraction", "sigma_u"),
    ]

    headers = ["Metric", "Symbol", "E1", "E2", "E3", "E4", "E5"]
    table: List[List[str]] = [headers]
    for metric_name, symbol in rows:
        row = [metric_name, symbol]
        for exp_key in EXPERIMENT_ORDER:
            value = metrics_by_exp[exp_key][symbol]
            row.append(f"{value:.6f}")
        table.append(row)

    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(table):
            f.write("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + "\n")
            if idx == 0:
                f.write("  ".join("-" * width for width in widths) + "\n")


def make_report_plots(experiments: Dict[str, ExperimentData]) -> None:
    ensure_dir(FIGURES_DIR)

    plot_3d_trajectory(
        experiments,
        FIGURES_DIR / "path_xyz_3d",
        "Three-dimensional trajectory comparison of E1-E5 against the reference helix.",
    )

    for idx, axis_name in enumerate(("x", "y", "z")):
        plot_scalar_comparison(
            experiments,
            FIGURES_DIR / f"tracking_{axis_name}",
            f"{axis_name.upper()} position tracking for all experiments with the reference trajectory.",
            r"Position [m]",
            actual_getter=lambda ds, i=idx: ds["x"][:, i],
            reference_getter=lambda i=idx: (
                lambda t_ref, y_ref: (t_ref, y_ref[:, i])
            )(*first_reference(experiments, "x_des")),
            figsize=(7.4, 2.45),
        )

    for idx, axis_name in enumerate(("x", "y", "z")):
        plot_tracking_component(
            experiments,
            FIGURES_DIR / f"tracking_velocity_{axis_name}",
            f"{axis_name.upper()} velocity tracking for all experiments with the reference trajectory.",
            actual_key="v",
            desired_key="v_des",
            component_index=idx,
            ylabel=r"Velocity [m/s]",
        )

    for idx, axis_name in enumerate(("x", "y", "z")):
        plot_tracking_component(
            experiments,
            FIGURES_DIR / f"tracking_acceleration_{axis_name}",
            f"{axis_name.upper()} acceleration tracking for all experiments with the reference trajectory.",
            actual_key="a",
            desired_key="a_des",
            component_index=idx,
            ylabel=r"Acceleration [m/s$^2$]",
        )

    plot_scalar_comparison(
        experiments,
        FIGURES_DIR / "error_position",
        "Position-error norm comparison across E1-E5.",
        r"Position error [m]",
        actual_getter=lambda ds: safe_norm_rows(ds["e_x"]),
        reference_getter=None,
        legend_ncol=2,
        figsize=(7.4, 2.45),
    )

    plot_scalar_comparison(
        experiments,
        FIGURES_DIR / "error_velocity",
        "Velocity-error norm comparison across E1-E5.",
        r"Velocity error [m/s]",
        actual_getter=lambda ds: safe_norm_rows(ds["e_v"]),
        reference_getter=None,
        legend_ncol=2,
        figsize=(7.4, 2.45),
    )

    for idx, axis_name in enumerate(("x", "y", "z")):
        plot_scalar_comparison(
            experiments,
            FIGURES_DIR / f"error_velocity_{axis_name}",
            f"{axis_name.upper()} velocity tracking error for all experiments.",
            r"Velocity error [m/s]",
            actual_getter=lambda ds, i=idx: ds["e_v"][:, i],
            reference_getter=None,
            legend_ncol=2,
            figsize=(7.4, 2.45),
        )

    for idx, axis_name in enumerate(("x", "y", "z")):
        plot_scalar_comparison(
            experiments,
            FIGURES_DIR / f"error_acceleration_{axis_name}",
            f"{axis_name.upper()} acceleration tracking error for all experiments.",
            r"Acceleration error [m/s$^2$]",
            actual_getter=lambda ds, i=idx: ds["a"][:, i] - ds["a_des"][:, i],
            reference_getter=None,
            legend_ncol=2,
            figsize=(7.4, 2.45),
        )

    plot_scalar_comparison(
        experiments,
        FIGURES_DIR / "error_attitude",
        "Attitude-error norm comparison across E1-E5.",
        r"Attitude error",
        actual_getter=lambda ds: safe_norm_rows(ds["e_R"]),
        reference_getter=None,
        legend_ncol=2,
        figsize=(7.4, 2.45),
    )

    plot_scalar_comparison(
        experiments,
        FIGURES_DIR / "error_angular_velocity",
        "Angular-velocity-error norm comparison across E1-E5.",
        r"$\|e_{\Omega}\|$ [rad/s]",
        actual_getter=lambda ds: safe_norm_rows(ds["e_omega"]),
        reference_getter=None,
        legend_ncol=2,
        figsize=(7.4, 2.45),
    )

    for idx, axis_name in enumerate(("x", "y", "z")):
        plot_tracking_component(
            experiments,
            FIGURES_DIR / f"tracking_angular_acceleration_{axis_name}",
            f"{axis_name.upper()} angular-acceleration tracking for all experiments with the reference trajectory.",
            actual_key="alpha",
            desired_key="alpha_des",
            component_index=idx,
            ylabel=rf"$\alpha_{{{axis_name}}}$ [rad/s$^2$]",
        )

    for idx, axis_name in enumerate(("x", "y", "z")):
        plot_scalar_comparison(
            experiments,
            FIGURES_DIR / f"error_angular_acceleration_{axis_name}",
            f"{axis_name.upper()} angular-acceleration tracking error for all experiments.",
            rf"$e_{{\alpha,{axis_name}}}$ [rad/s$^2$]",
            actual_getter=lambda ds, i=idx: ds["alpha"][:, i] - ds["alpha_des"][:, i],
            reference_getter=None,
            legend_ncol=2,
            figsize=(7.4, 2.45),
            t_min=1.0,
        )

    plot_scalar_comparison(
        experiments,
        FIGURES_DIR / "error_angular_acceleration",
        "Angular-acceleration-error norm comparison across E1-E5.",
        r"$\|e_{\alpha}\|$ [rad/s$^2$]",
        actual_getter=lambda ds: safe_norm_rows(ds["alpha"] - ds["alpha_des"]),
        reference_getter=None,
        legend_ncol=2,
        figsize=(7.4, 2.45),
        t_min=1.0,
    )

    plot_scalar_comparison(
        experiments,
        FIGURES_DIR / "control_Mx",
        "Commanded roll moment comparison across E1-E5 after the first second.",
        r"$M_x$ [N m]",
        actual_getter=lambda ds: ds["M"][:, 0],
        reference_getter=None,
        legend_ncol=2,
        figsize=(7.4, 2.45),
        t_min=1.0,
    )

    plot_scalar_comparison(
        experiments,
        FIGURES_DIR / "control_My",
        "Commanded pitch moment comparison across E1-E5 after the first second.",
        r"$M_y$ [N m]",
        actual_getter=lambda ds: ds["M"][:, 1],
        reference_getter=None,
        legend_ncol=2,
        figsize=(7.4, 2.45),
        t_min=1.0,
    )

    plot_scalar_comparison(
        experiments,
        FIGURES_DIR / "control_Mz",
        "Commanded yaw moment comparison across E1-E5 after the first second.",
        r"$M_z$ [N m]",
        actual_getter=lambda ds: ds["M"][:, 2],
        reference_getter=None,
        legend_ncol=2,
        figsize=(7.4, 2.45),
        t_min=1.0,
    )

    plot_scalar_comparison(
        experiments,
        FIGURES_DIR / "control_f",
        "Commanded total thrust comparison across E1-E5 after the first second.",
        r"$f$ [N]",
        actual_getter=lambda ds: ds["f"],
        reference_getter=None,
        legend_ncol=2,
        figsize=(7.4, 2.45),
        t_min=1.0,
    )

    plot_throttle_components(experiments, FIGURES_DIR)


def make_appendix_plots(experiments: Dict[str, ExperimentData]) -> None:
    ensure_dir(FIGURES_DIR)

    for idx, axis_name in enumerate(("x", "y", "z")):
        plot_scalar_comparison(
            experiments,
            FIGURES_DIR / f"error_{axis_name}",
            f"{axis_name.upper()} position tracking error for all experiments.",
            r"Position error [m]",
            actual_getter=lambda ds, i=idx: ds["e_x"][:, i],
            reference_getter=None,
            legend_ncol=2,
            figsize=(7.4, 2.45),
        )

    for idx, (name, ylabel) in enumerate((
        ("roll", r"Roll [deg]"),
        ("pitch", r"Pitch [deg]"),
        ("yaw", r"Yaw [deg]"),
    )):
        if name == "yaw":
            plot_tracking_component_with_transform(
                experiments,
                FIGURES_DIR / f"tracking_attitude_{name}",
                f"{name.capitalize()} tracking across E1-E5 and the reference attitude signal.",
                actual_key="rpy",
                desired_key="rpy_des",
                component_index=idx,
                ylabel=ylabel,
                actual_transform=lambda x: wrap_deg_180(np.rad2deg(x)),
                desired_transform=lambda x: wrap_deg_180(np.rad2deg(x)),
            )
        else:
            plot_tracking_component_with_transform(
                experiments,
                FIGURES_DIR / f"tracking_attitude_{name}",
                f"{name.capitalize()} tracking across E1-E5 and the reference attitude signal.",
                actual_key="rpy",
                desired_key="rpy_des",
                component_index=idx,
                ylabel=ylabel,
                actual_transform=np.rad2deg,
                desired_transform=np.rad2deg,
            )

    for idx, axis_name in enumerate(("x", "y", "z")):
        plot_tracking_component(
            experiments,
            FIGURES_DIR / f"tracking_omega_{axis_name}",
            f"Angular-rate tracking in {axis_name} across E1-E5 and the reference signal.",
            actual_key="omega",
            desired_key="omega_des",
            component_index=idx,
            ylabel=rf"$\Omega_{{{axis_name}}}$ [rad/s]",
        )

    for idx, name in enumerate(("x", "y", "z")):
        plot_error_components(
            experiments,
            FIGURES_DIR / f"error_attitude_component_{name}",
            f"Attitude-error component {name} across E1-E5.",
            error_getter=lambda ds: ds["e_R"],
            ylabel=rf"Attitude error component {name}",
            component_index=idx,
            component_label=rf"$e_{{R,{idx + 1}}}$",
        )

    for idx, name in enumerate(("x", "y", "z")):
        plot_error_components(
            experiments,
            FIGURES_DIR / f"error_angular_velocity_component_{name}",
            f"Angular-velocity-error component {name} across E1-E5.",
            error_getter=lambda ds: ds["e_omega"],
            ylabel=rf"$e_{{\Omega,{name}}}$ [rad/s]",
            component_index=idx,
            component_label=rf"$e_{{\Omega,{idx + 1}}}$",
        )

    write_metrics_table(experiments, METRICS_PATH)
    SAVED_OUTPUTS.append((
        METRICS_PATH,
        "Text table of aggregate tracking metrics for E1-E5.",
        (0.0, 0.0),
    ))


def print_saved_outputs() -> None:
    print("\nSaved outputs:")
    for path, description, size_inches in SAVED_OUTPUTS:
        print(f"- {relative_figure_path(path)}")
        print(f"  {description}")
        if path.suffix == ".png":
            print(f"  Figure size: {size_inches[0]:.2f} in x {size_inches[1]:.2f} in")


def main() -> int:
    ensure_dir(FIGURES_DIR)
    experiments = load_experiments()
    normalize_throttles(experiments)

    if metrics_only_requested():
        write_metrics_table(experiments, METRICS_PATH)
        SAVED_OUTPUTS.append((
            METRICS_PATH,
            "Text table of aggregate tracking metrics for E1-E5.",
            (0.0, 0.0),
        ))
        print(f"Updated metrics only under: {FIGURES_DIR}")
        print_saved_outputs()
        return 0

    setup_matplotlib()
    make_report_plots(experiments)
    make_appendix_plots(experiments)
    print(f"Saved figures under: {FIGURES_DIR}")
    print_saved_outputs()
    return 0


if __name__ == "__main__":
    sys.exit(main())
