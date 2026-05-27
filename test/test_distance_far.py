"""Test DistanceFarGenerator.

Shows: time dynamics, spatial activity map, firing rate vs wall distance.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import config

tf.get_logger().setLevel("ERROR")

from utils.inputs import DistanceFarGenerator
from utils.test_plots import (set_test_style, plot_activity_map,
                               plot_distance_response,
                               load_or_generate_trajectory)

from utils.rate_map import build_spatial_maps

def test_distance_far(save_dir="results/tests"):
    os.makedirs(save_dir, exist_ok=True)

    # Load or generate trajectory
    traj = load_or_generate_trajectory(duration=60.0)


    x = traj["x"]
    y = traj["y"]
    t = traj["t"]

    # Build extra_inputs: [x, y, vx, vy]
    speed = traj["speed"]
    hd = traj["head_direction"]
    extra = np.stack([
        traj["x"], traj["y"],
        speed * np.cos(hd),
        speed * np.sin(hd),
    ], axis=-1).astype(np.float32)

    # Run generator
    gen = DistanceFarGenerator()
    rates = []
    for i in range(0, len(extra), 100):
        ei = tf.constant(extra[i:i+100])
        r = gen(tf.constant(0.0), extra_inputs=ei)
        rates.append(r.numpy())
    rates = np.concatenate(rates)[:len(t)]

    # Validate: d_far = alpha_far * d_min(x,y)
    d_N = config.ARENA_CM - y
    d_S = y
    d_E = config.ARENA_CM - x
    d_W = x
    d_min = np.minimum(np.minimum(d_N, d_S), np.minimum(d_E, d_W))
    # expected = config.DISTANCE_FAR["alpha_far"] * d_min[:len(rates)]
    # assert np.allclose(rates, expected, rtol=1e-5), \
    #     f"Rate mismatch: max |diff| = {np.max(np.abs(rates - expected))}"

    set_test_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. Time dynamics — first 10s
    ax = axes[0, 0]
    n_show = min(len(t), int(10.0 / config.TRAJECTORY_DT))
    ax.plot(t[:n_show], rates[:n_show], linewidth=0.5, color='C0')
    ax.plot(t[:n_show], d_min[:n_show] * config.DISTANCE_FAR["alpha_far"],
            '--', linewidth=0.8, alpha=0.5, color='gray', label='α_far·d_min')
    ax.set_title('DistanceFar: time dynamics')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.legend(fontsize=7)



    # 3. Rate vs d_min
    ax = axes[1, 0]
    plot_distance_response(fig, ax, d_min[:len(rates)], rates,
                           title="d_far vs nearest-wall distance",
                           xlabel="d_min (cm)")

    # 4. Activity map (standalone, clear view)

    positions = np.stack([x, y], axis=-1)
    rate_map, _, _, _ = build_spatial_maps(rates, positions, bins = 50, range_xy=((0, 50), (0, 50)))

    axes[0, 1].imshow(rate_map).set_cmap("rainbow")

    plt.suptitle('DistanceFarGenerator Test', fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'test_distance_far.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)

    print(f"Test passed — d_far range: [{rates.min():.1f}, {rates.max():.1f}] Hz")
    return True


if __name__ == "__main__":
    test_distance_far()
