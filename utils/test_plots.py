"""Common plotting utilities for generator tests."""

import numpy as np
import matplotlib.pyplot as plt
import config


def set_test_style():
    plt.style.use('default')
    plt.rcParams.update({
        'figure.dpi': 120, 'font.size': 9,
        'axes.titlesize': 11, 'axes.labelsize': 10,
    })


def plot_activity_map(x, y, rate, title="Activity map",
                      cmap="viridis", vmin=0.0, save_path=None):
    """Plot firing rate as colour on arena coordinates.

    Args:
        x, y: position arrays (cm), shape (n,)
        rate: firing rate array (Hz), shape (n,)
        title: plot title
        cmap: colormap name
        vmin: colour scale minimum
        save_path: if set, save figure to this path
    """
    set_test_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    step = max(1, len(x) // 8000)
    idx = slice(0, len(x), step)
    sc = ax.scatter(x[idx], y[idx], c=rate[idx], cmap=cmap,
                    s=1.5, alpha=0.7, vmin=vmin)
    cbar = plt.colorbar(sc, ax=ax, label='Hz', shrink=0.8)
    ax.set_xlim(0, config.ARENA_CM)
    ax.set_ylim(0, config.ARENA_CM)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, ax


def load_or_generate_trajectory(duration=60.0):
    """Load trajectory from HDF5, or generate if not found.

    Returns a dict with keys: x, y, speed, head_direction,
    d_N, d_S, d_E, d_W, d_min, t.
    """
    import os
    dt = config.TRAJECTORY_DT
    n_desired = int(duration / dt)

    if os.path.exists(config.TRAJECTORY_HDF5):
        print(f"Loading trajectory from {config.TRAJECTORY_HDF5}")

        import h5py
        traj = {}
        with h5py.File(config.TRAJECTORY_HDF5, 'r') as f:
            for k in f.keys():
                data = f[k][:]
                n = min(len(data), n_desired)
                traj[k] = data[:n]
        # HDF5 speed is [vx, vy] — compute scalar speed and head_direction
        vx = traj['speed'][:, 0]
        vy = traj['speed'][:, 1]
        traj['head_direction'] = np.arctan2(vy, vx)
        traj['speed'] = np.sqrt(vx**2 + vy**2)
        traj['t'] = np.arange(n, dtype=np.float32) * dt
        return traj
    else:
        print("Generating trajectory")
        from utils.trajectory import TrajectoryGenerator
        gen = TrajectoryGenerator(seed=config.RANDOM_SEED)

        traj = gen.generate(duration)

        vx = traj['speed'][:, 0]
        vy = traj['speed'][:, 1]
        traj['head_direction'] = np.arctan2(vy, vx)
        traj['speed'] = np.sqrt(vx**2 + vy**2)
        traj['t'] = np.arange(len(traj['head_direction']), dtype=np.float32) * dt
        return traj


def generate_trajectory_for_test(duration=60.0):
    """Generate a fresh trajectory for testing (ignores HDF5)."""
    from utils.trajectory import TrajectoryGenerator
    gen = TrajectoryGenerator(seed=config.RANDOM_SEED)
    return gen.generate(duration)

# ============================================================
# Common generator test function for distance-based generators
# ============================================================

def plot_distance_response(fig, ax, distances, rates, xlabel="Distance (cm)",
                           title="Firing rate vs distance"):
    """Plot firing rate as a function of distance."""
    ax.scatter(distances[::5], rates[::5], s=1, alpha=0.3, color='C0')
    # Bin and average
    bins = np.linspace(0, max(distances), 50)
    digitized = np.digitize(distances, bins)
    bin_means = np.array([rates[digitized == i].mean()
                          for i in range(1, len(bins)) if np.any(digitized == i)])
    bin_centres = np.array([(bins[i] + bins[i-1]) / 2
                            for i in range(1, len(bins)) if np.any(digitized == i)])
    ax.plot(bin_centres, bin_means, 'k-', linewidth=2, label='Mean')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_speed_response(fig, ax, speeds, rates, xlabel="Speed (cm/s)",
                        title="Firing rate vs speed"):
    """Plot firing rate as a function of absolute speed."""
    ax.scatter(speeds[::5], rates[::5], s=1, alpha=0.3, color='C1')
    bins = np.linspace(0, max(speeds), 40)
    digitized = np.digitize(speeds, bins)
    bin_means = np.array([rates[digitized == i].mean()
                          for i in range(1, len(bins)) if np.any(digitized == i)])
    bin_centres = np.array([(bins[i] + bins[i-1]) / 2
                            for i in range(1, len(bins)) if np.any(digitized == i)])
    ax.plot(bin_centres, bin_means, 'k-', linewidth=2, label='Mean')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_hd_response(fig, ax, head_angles, rates, xlabel="Head direction (deg)",
                     title="Firing rate vs head direction"):
    """Plot firing rate as a function of head direction."""
    hd_deg = np.rad2deg(head_angles)
    ax.scatter(hd_deg[::5], rates[::5], s=1, alpha=0.3, color='C2')
    bins = np.linspace(-180, 180, 72)
    digitized = np.digitize(hd_deg, bins)
    bin_means = np.array([rates[digitized == i].mean()
                          for i in range(1, len(bins)) if np.any(digitized == i)])
    bin_centres = np.array([(bins[i] + bins[i-1]) / 2
                            for i in range(1, len(bins)) if np.any(digitized == i)])
    ax.plot(bin_centres, bin_means, 'k-', linewidth=2, label='Mean')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
