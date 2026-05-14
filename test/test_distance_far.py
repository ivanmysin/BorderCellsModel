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


def test_distance_far(save_dir="results/tests"):
    os.makedirs(save_dir, exist_ok=True)

    # Load or generate trajectory
    traj = load_or_generate_trajectory(duration=60.0)
    t = traj["t"]
    x = traj["x"]
    y = traj["y"]
    d_min = traj["d_min"]

    # Build extra_inputs: [d_min, speed, cos_hd, sin_hd, d_N, d_S, d_E, d_W]
    extra = np.stack([
        traj["d_min"], traj["speed"],
        np.cos(traj["head_direction"]), np.sin(traj["head_direction"]),
        traj["d_N"], traj["d_S"], traj["d_E"], traj["d_W"],
    ], axis=-1).astype(np.float32)

    # Run generator
    gen = DistanceFarGenerator()
    rates = []
    for i in range(0, len(extra), 100):
        ei = tf.constant(extra[i:i+100])
        r = gen(tf.constant(0.0), extra_inputs=ei)
        rates.append(r.numpy())
    rates = np.concatenate(rates)[:len(t)]

    # Validate: d_far = alpha_far * d_min
    expected = config.DISTANCE_FAR["alpha_far"] * d_min[:len(rates)]
    assert np.allclose(rates, expected, rtol=1e-5), \
        f"Rate mismatch: max |diff| = {np.max(np.abs(rates - expected))}"

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

    # 2. Activity map
    plt.sca(axes[0, 1])
    idx = min(len(x), len(rates))
    plot_activity_map(
        x[:idx], y[:idx], rates[:idx],
        title="DistanceFar: spatial activity", cmap="Reds",
    )
    plt.sca(axes[0, 0])  # restore

    # 3. Rate vs d_min
    ax = axes[1, 0]
    idx = min(len(d_min), len(rates))
    plot_distance_response(fig, ax, d_min[:idx], rates[:idx],
                           title="d_far vs nearest-wall distance",
                           xlabel="d_min (cm)")

    # 4. Target: d_far should increase linearly with d_min
    ax = axes[1, 1]
    from utils.dataset import compute_targets
    targets = compute_targets(traj)
    ax.plot(t[:n_show], rates[:n_show], linewidth=0.5, color='C0',
            label='d_far (far-active)')
    # d_near = alpha_near * (D_max - d_min)
    near_rate = config.DISTANCE_NEAR["alpha_near"] * (
        config.DISTANCE_NEAR["d_max"] - d_min[:n_show])
    ax.plot(t[:n_show], np.maximum(near_rate, 0), linewidth=0.5,
            color='C3', alpha=0.5, label='d_near (near-active)')
    ax.set_title('Distance channels comparison')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.legend(fontsize=7)

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
