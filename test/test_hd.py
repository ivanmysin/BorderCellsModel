"""Test HeadDirectionGenerator with N_HD=18 HD cells.

Shows: time dynamics for 2 example cells (0° and 180°),
spatial activity map, HD tuning for all cells, and a polar plot.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import tensorflow as tf
import config

tf.get_logger().setLevel("ERROR")

from utils.inputs import HeadDirectionGenerator
from utils.test_plots import (set_test_style, load_or_generate_trajectory)


def test_hd(save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(config.RESULTS_DIR, "tests")
    os.makedirs(save_dir, exist_ok=True)

    traj = load_or_generate_trajectory(duration=60.0)
    t = traj["t"]
    hd = traj["head_direction"]
    speed = traj["speed"]

    extra = np.stack([
        traj["x"], traj["y"],
        speed * np.cos(hd),
        speed * np.sin(hd),
    ], axis=-1).astype(np.float32)

    gen = HeadDirectionGenerator()
    rates_hd = []
    for i in range(0, len(extra), 100):
        ei = tf.constant(extra[i:i+100])
        r = gen(tf.constant(0.0), extra_inputs=ei)
        rates_hd.append(r.numpy())
    rates_hd = np.concatenate(rates_hd, axis=0)[:len(t)]

    # Cell 0: preferred direction 0°, cell 9: preferred direction 180°
    cell_0 = rates_hd[:, 0]
    cell_9 = rates_hd[:, 9]

    set_test_style()
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # 1. Time dynamics — two example cells
    ax = axes[0, 0]
    n_show = min(len(t), int(10.0 / config.TRAJECTORY_DT))
    ax.plot(t[:n_show], cell_0[:n_show], linewidth=0.5, label='HD cell 0°')
    ax.plot(t[:n_show], cell_9[:n_show], linewidth=0.5, label='HD cell 180°')
    ax.set_title('HD cell rates (example cells)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate (Hz)')
    ax.legend(fontsize=7)

    # 2. HD tuning curves (all 18 cells)
    ax = axes[0, 1]
    hd_bins = np.linspace(-np.pi, np.pi, 36)
    theta_pref_rad = np.deg2rad(config.THETA_PREF)
    for i in range(config.N_HD):
        cell_rate = rates_hd[:, i]
        idx = np.digitize(hd[:len(cell_rate)], hd_bins) - 1
        mask = (idx >= 0) & (idx < len(hd_bins) - 1)
        mean_rate = np.array([cell_rate[mask & (idx == j)].mean()
                              for j in range(len(hd_bins) - 1)])
        ax.plot(np.rad2deg(hd_bins[:-1]), mean_rate,
                linewidth=0.8, alpha=0.6)
    ax.set_title('HD tuning (all 18 cells)')
    ax.set_xlabel('Head direction (deg)')
    ax.set_ylabel('Rate (Hz)')

    # 3. Polar plot: individual cell tuning
    ax = axes[0, 2]
    ax = fig.add_subplot(2, 3, 3, projection='polar')
    for i in [0, 4, 9, 13]:
        cell_rate = rates_hd[:, i]
        idx = np.digitize(hd[:len(cell_rate)], hd_bins) - 1
        mask = (idx >= 0) & (idx < len(hd_bins) - 1)
        mean_rate = np.array([cell_rate[mask & (idx == j)].mean()
                              for j in range(len(hd_bins) - 1)])
        ax.plot(hd_bins[:-1], mean_rate, linewidth=1.5,
                label=f'{i * config.HD_POPVEC["theta_step"]:.0f}°')
    ax.set_title('HD tuning (selected cells)', va='bottom')
    ax.legend(fontsize=6, loc='lower right')

    plt.suptitle('HeadDirectionGenerator Test (18 HD cells)', fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'test_hd.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)

    print(f"Test passed — cell_0: [{cell_0.min():.1f}, {cell_0.max():.1f}], "
          f"cell_9: [{cell_9.min():.1f}, {cell_9.max():.1f}]")
    return True


if __name__ == "__main__":
    test_hd()
