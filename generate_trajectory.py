"""Generate agent trajectories in a 1×1 m arena and save to HDF5.

Usage:
    python3 generate_trajectory.py [--plot] [--duration 10] [--n-trials 180] [--seed 42]

By default generates N_TRIALS=180 short trajectories of TRIAL_DURATION=10 s each
(per config.py), concatenated into a single trajectory stream. The simulation
in train_simple.py then splits this stream into BATCH_DURATION=0.1 s chunks and
feeds them sequentially to a stateful RNN so the network's internal state
propagates across batches.

Pass --n-trials 1 to fall back to the legacy single-trajectory mode.
"""
import os
import argparse
import numpy as np
import h5py
import config
from utils.trajectory import (
    TrajectoryGenerator, interpolate_trajectory,
    generate_concatenated_trajectories,
)


def generate_and_save(duration: float, plot: bool = False, n_trials: int = 1):
    """Generate trajectory and save to HDF5."""

    if config.RANDOM_SEED:
        seed = config.RANDOM_SEED
    else:
        seed = None

    gen = TrajectoryGenerator(seed)

    if n_trials == 1:
        traj_coarse = gen.generate(duration)
    else:
        total = n_trials * duration
        print(f"Generating {n_trials} trials × {duration:.1f} s = {total:.1f} s total...")
        traj_coarse = generate_concatenated_trajectories(gen, duration, n_trials)

    target_dt = config.DT / 1000.0
    traj = interpolate_trajectory(traj_coarse, target_dt)

    os.makedirs(os.path.dirname(config.TRAJECTORY_HDF5), exist_ok=True)

    with h5py.File(config.TRAJECTORY_HDF5, "w") as f:
        for key, arr in traj.items():
            f.create_dataset(key, data=arr.astype(np.float32))

    print(f"Trajectory saved to {config.TRAJECTORY_HDF5}")
    n_steps = len(traj["x"])
    if n_trials == 1:
        print(f"  Duration: {duration:.1f} s, Steps: {n_steps}")
    else:
        print(f"  Trials: {n_trials} × {duration:.1f} s = {n_trials * duration:.1f} s")
        print(f"  Steps: {n_steps}")
    print(f"  dt: {config.DT:.3f} ms")
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
    t = np.arange( len(traj["x"]) ) * config.TRAJECTORY_DT

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

    speed = np.sqrt(traj["vx"]**2 + traj["vy"]**2)
    head_direction = np.arctan2(traj["vy"], traj["vx"])

    ax = axes[1, 0]
    ax.plot(t, speed, linewidth=0.5, color="C1")
    ax.set_title("Speed")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (cm/s)")

    # Head direction
    ax = axes[1, 1]
    ax.plot(t, np.rad2deg(head_direction), linewidth=0.5, color="C3")
    ax.set_title("Head direction")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")

    plt.suptitle(f"Trajectory overview ({t[-1]:.0f}s)", fontsize=14)
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
    parser.add_argument("--duration", type=float, default=config.TRIAL_DURATION,
                        help=f"Trajectory duration per trial in seconds "
                             f"(default: {config.TRIAL_DURATION})")
    parser.add_argument("--n-trials", type=int, default=config.N_TRIALS,
                        help=f"Number of trials to generate and concatenate "
                             f"(default: {config.N_TRIALS})")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Plot trajectory overview")
    args = parser.parse_args()

    generate_and_save(args.duration, plot=args.plot, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
