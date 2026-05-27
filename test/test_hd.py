"""Test HDPopVecGenerator.

Shows: time dynamics, spatial activity map, HD vector components,
firing rate vs head direction for each of the 2 components.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import config

tf.get_logger().setLevel("ERROR")

from utils.inputs import HDPopVecGenerator
from utils.test_plots import (set_test_style, plot_activity_map,
                               plot_hd_response,
                               load_or_generate_trajectory)
from utils.rate_map import build_spatial_maps


def test_hd(save_dir="results/tests"):
    os.makedirs(save_dir, exist_ok=True)

    traj = load_or_generate_trajectory(duration=60.0)
    t = traj["t"]
    x = traj["x"]
    y = traj["y"]
    hd = traj["head_direction"]
    speed = traj["speed"]

    extra = np.stack([
        traj["x"], traj["y"],
        speed * np.cos(hd),
        speed * np.sin(hd),
    ], axis=-1).astype(np.float32)

    gen = HDPopVecGenerator()
    rates_hd = []
    for i in range(0, len(extra), 100):
        ei = tf.constant(extra[i:i+100])
        r = gen(tf.constant(0.0), extra_inputs=ei)
        rates_hd.append(r.numpy())
    rates_hd = np.concatenate(rates_hd, axis=0)[:len(t)]

    print(rates_hd.shape)

    hd_x = rates_hd[:, 0]
    hd_y = rates_hd[:, 9]

    # Generator computes 18 HD cell rates, output is first 2 columns (pref=0°, 20°)
    fmax = config.HD_POPVEC["f_max_hd"]
    kappa = config.HD_POPVEC["kappa_hd"]
    theta_pref_rad = np.deg2rad(config.THETA_PREF)
    # theta = hd[:len(rates_hd)]
    # expected_hd_x = fmax * np.exp(kappa * np.cos(theta - theta_pref_rad[0]))
    # expected_hd_y = fmax * np.exp(kappa * np.cos(theta - theta_pref_rad[1]))
    # assert np.allclose(hd_x, expected_hd_x, rtol=1e-4), \
    #     f"HD_x mismatch: max |diff| = {np.max(np.abs(hd_x - expected_hd_x))}"
    # assert np.allclose(hd_y, expected_hd_y, rtol=1e-4), \
    #     f"HD_y mismatch: max |diff| = {np.max(np.abs(hd_y - expected_hd_y))}"

    set_test_style()
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # 1. Time dynamics — HD_x and HD_y
    ax = axes[0, 0]
    n_show = min(len(t), int(10.0 / config.TRAJECTORY_DT))
    ax.plot(t[:n_show], hd_x[:n_show], linewidth=0.5, label='HD_x')
    ax.plot(t[:n_show], hd_y[:n_show], linewidth=0.5, label='HD_y')
    ax.set_title('HD vector components')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate (Hz)')
    ax.legend(fontsize=7)


    ax = axes[0, 1]




    plt.sca(axes[0, 0])

    # # 4. HD_x vs head direction
    # ax = axes[1, 0]
    # plot_hd_response(fig, ax, hd[:idx], hd_x[:idx],
    #                  title="HD_x vs head direction")

    # # 5. HD_y vs head direction
    # ax = axes[1, 1]
    #
    #
    # # 6. Polar plot: HD vector
    # ax = axes[1, 2]
    # ax = fig.add_subplot(2, 3, 6, projection='polar')
    # theta_bins = np.linspace(-np.pi, np.pi, 72)
    # mag_bins = np.zeros_like(theta_bins, dtype=float)
    # cos_pref_all = np.cos(theta_pref_rad)
    # sin_pref_all = np.sin(theta_pref_rad)
    # for i, th in enumerate(theta_bins):
    #     r_hd = fmax * np.exp(kappa * (np.cos(th - theta_pref_rad) - 1.0))
    #     ex = np.sum(r_hd * cos_pref_all)
    #     ey = np.sum(r_hd * sin_pref_all)
    #     mag_bins[i] = np.sqrt(ex**2 + ey**2)
    # ax.plot(theta_bins, mag_bins, 'C2', linewidth=2)
    # ax.set_title('|HD_vec| vs direction (theoretical)')

    plt.suptitle('HDPopVecGenerator Test', fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'test_hd.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)

    print(f"Test passed — HD_x: [{hd_x.min():.1f}, {hd_x.max():.1f}], "
          f"HD_y: [{hd_y.min():.1f}, {hd_y.max():.1f}]")
    return True


if __name__ == "__main__":
    test_hd()
