"""Generate agent trajectories in a 1×1 m arena and save to HDF5.

Usage:
    python3 generate_trajectory.py [--plot] [--duration 600] [--seed 42]
"""

import os
import argparse
import numpy as np
import h5py
import config
from utils.trajectory import TrajectoryGenerator


def generate_and_save(duration: float, plot: bool = False):
    """Generate trajectory and save to HDF5."""

    if config.RANDOM_SEED:
        seed = config.RANDOM_SEED
    else:
        seed = None

    gen = TrajectoryGenerator(seed)
    traj = gen.generate(duration)

    os.makedirs(os.path.dirname(config.TRAJECTORY_HDF5), exist_ok=True)

    with h5py.File(config.TRAJECTORY_HDF5, "w") as f:
        for key, arr in traj.items():
            f.create_dataset(key, data=arr.astype(np.float32), compression="gzip")

    print(f"Trajectory saved to {config.TRAJECTORY_HDF5}")
    n_steps = len(traj["t"])
    print(f"  Duration: {duration:.1f} s, Steps: {n_steps}")
    print(f"  dt: {config.TRAJECTORY_DT:.0f} ms")
    print(f"  d_min range: [{traj['d_min'].min():.1f}, {traj['d_min'].max():.1f}] cm")
    print(f"  x range: [{traj['x'].min():.1f}, {traj['x'].max():.1f}] cm")
    print(f"  y range: [{traj['y'].min():.1f}, {traj['y'].max():.1f}] cm")
    print(f"  Speed range: [{traj['speed'].min():.1f}, {traj['speed'].max():.1f}] cm/s")

    if plot:
        _plot_trajectory(traj)

    return traj


def _plot_trajectory(traj: dict):
    """Plot trajectory overview."""
    import matplotlib.pyplot as plt

    x = traj["x"][::10]
    y = traj["y"][::10]
    t = traj["t"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Arena trajectory
    ax = axes[0, 0]
    ax.plot(x, y, color="gray", linewidth=0.3, alpha=0.5)
    ax.set_xlim(0, config.ARENA_CM)
    ax.set_ylim(0, config.ARENA_CM)
    ax.set_aspect("equal")
    ax.set_title(f"Arena trajectory ({config.ARENA_CM}×{config.ARENA_CM}) cm)")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")

    # Distance to walls over time
    ax = axes[0, 1]
    ax.plot(t, traj["d_N"], linewidth=0.5, label="d_North")
    ax.plot(t, traj["d_S"], linewidth=0.5, label="d_South")
    ax.plot(t, traj["d_E"], linewidth=0.5, label="d_East")
    ax.plot(t, traj["d_W"], linewidth=0.5, label="d_West")
    ax.plot(t, traj["d_min"], "k", linewidth=1.0, label="d_min")
    ax.set_title("Distance to walls")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (cm)")
    ax.legend(fontsize=7)

    # Speed
    ax = axes[1, 0]
    ax.plot(t, traj["speed"], linewidth=0.5, color="C1")
    ax.set_title("Speed")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (cm/s)")

    # Head direction
    ax = axes[1, 1]
    ax.plot(t, np.rad2deg(traj["head_direction"]), linewidth=0.5, color="C3")
    ax.set_title("Head direction")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")

    plt.suptitle(f"Trajectory overview ({traj['t'][-1]:.0f}s)", fontsize=14)
    plt.tight_layout()
    save_path = config.RESULTS_DIR + "/trajectory_preview.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Trajectory plot saved to {save_path}")
    plt.close(fig)


def load_trajectory(path: str = None) -> dict:
    """Load trajectory from HDF5 file."""
    p = path or config.TRAJECTORY_HDF5
    traj = {}
    with h5py.File(p, "r") as f:
        for key in f.keys():
            traj[key] = f[key][:]
    return traj


def main():
    parser = argparse.ArgumentParser(description="Generate agent trajectory")
    parser.add_argument("--duration", type=float, default=600.0,
                        help="Trajectory duration in seconds (default: 600 = 10 min)")
    parser.add_argument("--plot", action="store_true", default=True,
                        help="Plot trajectory overview")
    args = parser.parse_args()

    generate_and_save(args.duration, plot=args.plot)


if __name__ == "__main__":
    main()
