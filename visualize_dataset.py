"""Visualize dataset: preview inputs, targets, and trajectory.

Usage:
    python visualize_dataset.py [--dataset data/dataset.h5] [--trajectory data/trajectory.h5]
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
from utils.dataset import load_trajectory_from_hdf5, load_dataset_hdf5


def visualize_dataset(dataset_path=None, trajectory_path=None,
                      output_dir=None, trial_idx=0):
    """Create visualization of the dataset."""

    ds_path = dataset_path or os.path.join(os.path.dirname(config.TRAJECTORY_HDF5), 'dataset.h5')
    traj_path = trajectory_path or config.TRAJECTORY_HDF5
    out_dir = output_dir or config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Load trajectory
    print(f"Loading trajectory from {traj_path}...")
    traj = load_trajectory_from_hdf5(traj_path)

    print(f"Loading dataset from {ds_path}...")
    ds = load_dataset_hdf5(ds_path)
    X = ds['X']
    Y = ds['Y']
    metadata = ds['metadata']

    if X.ndim != 3:
        raise ValueError(f"Expected X shape (n_trials, n_steps, 21), got {X.shape}")
    n_trials, n_steps, n_inputs = X.shape
    if n_inputs != config.N_INPUTS:
        raise ValueError(f"Expected {config.N_INPUTS} input channels, got {n_inputs}")

    dt_ms = float(metadata.get('dt', config.DT))
    t_ms = np.arange(n_steps, dtype=np.float32) * dt_ms
    t_s = t_ms / 1000.0

    if traj['x'].ndim == 2 and traj['x'].shape[0] != n_trials:
        print(f"  Warning: trajectory has {traj['x'].shape[0]} trials, dataset has {n_trials}")

    if trial_idx < 0 or trial_idx >= n_trials:
        print(f"  trial_idx {trial_idx} out of range [0, {n_trials}); falling back to 0")
        trial_idx = 0

    inputs = X[trial_idx]
    targets = Y[trial_idx]

    fig = plt.figure(figsize=(16, 14))

    # 1. Trajectory
    ax1 = fig.add_subplot(3, 2, 1)
    if traj['x'].ndim == 2:
        for i in range(traj['x'].shape[0]):
            ax1.plot(traj['x'][i], traj['y'][i], linewidth=0.3, alpha=0.5, color='gray')
    else:
        ax1.plot(traj['x'], traj['y'], linewidth=0.2, alpha=0.5, color='gray')
    ax1.set_xlim(0, config.ARENA_CM)
    ax1.set_ylim(0, config.ARENA_CM)
    ax1.set_aspect('equal')
    ax1.set_title(f'Trajectory ({config.ARENA_CM}x{config.ARENA_CM} cm, {n_trials} trials)')
    ax1.set_xlabel('x (cm)')
    ax1.set_ylabel('y (cm)')

    # 2. Input channels: d_far, d_near, speed
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(t_s, inputs[:, 0], label='d_far', linewidth=0.8)
    ax2.plot(t_s, inputs[:, 1], label='d_near', linewidth=0.8)
    ax2.plot(t_s, inputs[:, 2], label='speed', linewidth=0.8)
    ax2.set_title(f'Scalar inputs (trial {trial_idx})')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Rate (Hz)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. HD heatmap
    ax3 = fig.add_subplot(3, 2, 3)
    hd_data = inputs[:, 3:21].T
    step = max(1, hd_data.shape[1] // 1000)
    ax3.imshow(hd_data[:, ::step], aspect='auto', origin='lower',
               extent=[t_s[0], t_s[-1], 0, 18], cmap='hot')
    ax3.set_title(f'HD inputs (18 channels, trial {trial_idx})')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('HD channel')

    # 4. Targets
    ax4 = fig.add_subplot(3, 2, 4)
    wall_names = ['N', 'S', 'E', 'W']
    for j in range(4):
        ax4.plot(t_s, targets[:, j], label=f'Border_{wall_names[j]}', linewidth=0.8)
    ax4.set_title(f'Target rates (trial {trial_idx})')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Rate (Hz)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Target spatial heatmap
    ax5 = fig.add_subplot(3, 2, 5)
    if traj['x'].ndim == 2:
        x_show = traj['x'][trial_idx]
        y_show = traj['y'][trial_idx]
    else:
        x_show = traj['x']
        y_show = traj['y']
    n_show = min(len(x_show), targets.shape[0])
    sc = ax5.scatter(x_show[:n_show], y_show[:n_show], c=targets[:n_show, 0],
                     s=0.5, cmap='YlOrRd', vmin=0, vmax=config.F_MAX_BORDER)
    plt.colorbar(sc, ax=ax5, label='Border_N rate (Hz)')
    ax5.set_xlim(0, config.ARENA_CM)
    ax5.set_ylim(0, config.ARENA_CM)
    ax5.set_aspect('equal')
    ax5.set_title(f'Border_N target spatial (trial {trial_idx})')

    # 6. Info
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')
    info_text = (
        f"Dataset: {ds_path}\n"
        f"Trials: {n_trials}\n"
        f"Steps/trial: {n_steps}\n"
        f"DT: {dt_ms} ms\n"
        f"Arena: {config.ARENA_CM} cm\n"
        f"Inputs: {config.N_INPUTS} channels\n"
        f"Targets: 4 walls\n"
        f"\nShapes: X={X.shape}, Y={Y.shape}\n"
        f"\nInput ranges (all trials):\n"
        f"  d_far: [{X[..., 0].min():.2f}, {X[..., 0].max():.2f}] Hz\n"
        f"  d_near: [{X[..., 1].min():.2f}, {X[..., 1].max():.2f}] Hz\n"
        f"  speed: [{X[..., 2].min():.2f}, {X[..., 2].max():.2f}] Hz\n"
        f"  HD: [{X[..., 3:].min():.2f}, {X[..., 3:].max():.2f}] Hz\n"
        f"\nTarget ranges (all trials):\n"
        f"  [{Y.min():.2f}, {Y.max():.2f}] Hz"
    )
    ax6.text(0.1, 0.5, info_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='center', fontfamily='monospace')

    plt.suptitle('Dataset Preview', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(out_dir, 'dataset_preview.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--trajectory", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--trial", type=int, default=0,
                        help="Trial index to show in time-series panels (default: 0)")
    args = parser.parse_args()

    visualize_dataset(args.dataset, args.trajectory, args.output_dir, args.trial)


if __name__ == '__main__':
    main()
