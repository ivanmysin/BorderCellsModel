"""Test DistanceNearGenerator.

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

from utils.inputs import DistanceNearGenerator
from utils.test_plots import (set_test_style, plot_activity_map,
                               plot_distance_response,
                               load_or_generate_trajectory)


def test_distance_near(save_dir="results/tests"):
    os.makedirs(save_dir, exist_ok=True)

    traj = load_or_generate_trajectory(duration=60.0)
    t = traj["t"]
    x = traj["x"]
    y = traj["y"]
    d_min = traj["d_min"]

    extra = np.stack([
        traj["d_min"], traj["speed"],
        np.cos(traj["head_direction"]), np.sin(traj["head_direction"]),
        traj["d_N"], traj["d_S"], traj["d_E"], traj["d_W"],
    ], axis=-1).astype(np.float32)

    gen = DistanceNearGenerator()
    rates = []
    for i in range(0, len(extra), 100):
        ei = tf.constant(extra[i:i+100])
        r = gen(tf.constant(0.0), extra_inputs=ei)
        rates.append(r.numpy())
    rates = np.concatenate(rates)[:len(t)]

    # Validate formula: alpha_near * (d_max - d_min), clipped >= 0
    a = config.DISTANCE_NEAR["alpha_near"]
    dm = config.DISTANCE_NEAR["d_max"]
    expected = np.maximum(a * (dm - d_min[:len(rates)]), 0)
    assert np.allclose(rates, expected, rtol=1e-5), \
        f"Rate mismatch: max |diff| = {np.max(np.abs(rates - expected))}"

    set_test_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. Time dynamics
    ax = axes[0, 0]
    n_show = min(len(t), int(10.0 / config.TRAJECTORY_DT))
    ax.plot(t[:n_show], rates[:n_show], linewidth=0.5, color='C3')
    ax.plot(t[:n_show], expected[:n_show], '--', linewidth=0.8,
            alpha=0.5, color='gray', label='α_near·(D_max−d_min)')
    ax.set_title('DistanceNear: time dynamics')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.legend(fontsize=7)

    # 2. Activity map
    plt.sca(axes[0, 1])
    idx = min(len(x), len(rates))
    plot_activity_map(
        x[:idx], y[:idx], rates[:idx],
        title="DistanceNear: spatial activity", cmap="Blues",
    )
    plt.sca(axes[0, 0])

    # 3. Rate vs d_min
    ax = axes[1, 0]
    idx = min(len(d_min), len(rates))
    plot_distance_response(fig, ax, d_min[:idx], rates[:idx],
                           title="d_near vs nearest-wall distance",
                           xlabel="d_min (cm)")

    # 4. Comparison: max rate vs min distance
    ax = axes[1, 1]
    ax.plot(d_min[:n_show:10], rates[:n_show:10], '.', markersize=1,
            alpha=0.3, color='C3')
    ax.set_xlabel('d_min (cm)')
    ax.set_ylabel('d_near rate (Hz)')
    ax.set_title('Rate = f(d_min)')
    ax.grid(True, alpha=0.3)

    plt.suptitle('DistanceNearGenerator Test', fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'test_distance_near.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)

    print(f"Test passed — d_near range: [{rates.min():.1f}, {rates.max():.1f}] Hz")
    return True


if __name__ == "__main__":
    test_distance_near()
