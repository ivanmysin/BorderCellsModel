"""Test SpeedGenerator.

Shows: time dynamics, spatial activity map, firing rate vs speed.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import config

tf.get_logger().setLevel("ERROR")

from utils.inputs import SpeedGenerator
from utils.test_plots import (set_test_style, plot_activity_map,
                               plot_speed_response,
                               load_or_generate_trajectory)


def test_speed(save_dir="results/tests"):
    os.makedirs(save_dir, exist_ok=True)

    traj = load_or_generate_trajectory(duration=60.0)
    t = traj["t"]
    x = traj["x"]
    y = traj["y"]
    speed = traj["speed"]

    extra = np.stack([
        traj["d_min"], traj["speed"],
        np.cos(traj["head_direction"]), np.sin(traj["head_direction"]),
        traj["d_N"], traj["d_S"], traj["d_E"], traj["d_W"],
    ], axis=-1).astype(np.float32)

    gen = SpeedGenerator()
    rates = []
    for i in range(0, len(extra), 100):
        ei = tf.constant(extra[i:i+100])
        r = gen(tf.constant(0.0), extra_inputs=ei)
        rates.append(r.numpy())
    rates = np.concatenate(rates)[:len(t)]

    # Validate formula: beta_0 + beta_1 * speed
    expected = (config.SPEED_CELL["beta_0"]
                + config.SPEED_CELL["beta_1"] * speed[:len(rates)])
    assert np.allclose(rates, expected, rtol=1e-5), \
        f"Rate mismatch: max |diff| = {np.max(np.abs(rates - expected))}"

    set_test_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. Time dynamics
    ax = axes[0, 0]
    n_show = min(len(t), int(10.0 / config.TRAJECTORY_DT))
    ax.plot(t[:n_show], rates[:n_show], linewidth=0.5, color='C1')
    ax2 = ax.twinx()
    ax2.plot(t[:n_show], speed[:n_show], linewidth=0.3, color='gray', alpha=0.4)
    ax2.set_ylabel('Speed (cm/s)', color='gray')
    ax.set_title('Speed cell: time dynamics')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.legend(['Rate'], fontsize=7)

    # 2. Activity map
    plt.sca(axes[0, 1])
    idx = min(len(x), len(rates))
    plot_activity_map(
        x[:idx], y[:idx], rates[:idx],
        title="Speed cell: spatial activity", cmap="Greens",
    )
    plt.sca(axes[0, 0])

    # 3. Rate vs speed magnitude
    ax = axes[1, 0]
    idx = min(len(speed), len(rates))
    plot_speed_response(fig, ax, speed[:idx], rates[:idx],
                        title="Speed cell tuning")

    # 4. Speed histogram + rate overlay
    ax = axes[1, 1]
    ax.hist(speed, bins=60, density=True, alpha=0.4, color='gray')
    ax.set_xlabel('Speed (cm/s)')
    ax.set_ylabel('Density')
    ax.set_title('Speed distribution')
    ax.grid(True, alpha=0.3)

    plt.suptitle('SpeedGenerator Test', fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'test_speed.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)

    print(f"Test passed — rate range: [{rates.min():.1f}, {rates.max():.1f}] Hz")
    return True


if __name__ == "__main__":
    test_speed()
