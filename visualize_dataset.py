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


def visualize_dataset(dataset_path: str = None, trajectory_path: str = None,
                      output_dir: str = None):
    """Create visualization of the dataset."""

    ds_path = dataset_path or os.path.join(os.path.dirname(config.TRAJECTORY_HDF5), 'dataset.h5')
    traj_path = trajectory_path or config.TRAJECTORY_HDF5
    out_dir = output_dir or config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Load trajectory
    print(f"Loading trajectory from {traj_path}...")
    traj = load_trajectory_from_hdf5(traj_path)

    # Load dataset
    print(f"Loading dataset from {ds_path}...")
    ds = load_dataset_hdf5(ds_path)
    n_batches = ds['n_batches']
    batch0 = ds['get_batch'](0)
    ds['file'].close()

    all_inputs = batch0['inputs'][0]    # [T, 21]
    all_targets = batch0['targets'][0]  # [T, 4]
    t_ms = batch0['t_seq'][0, :, 0]     # [T]
    t_s = t_ms / 1000.0

    fig = plt.figure(figsize=(16, 14))

    # 1. Trajectory
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(traj['x'], traj['y'], linewidth=0.2, alpha=0.5, color='gray')
    ax1.set_xlim(0, config.ARENA_CM)
    ax1.set_ylim(0, config.ARENA_CM)
    ax1.set_aspect('equal')
    ax1.set_title(f'Trajectory ({config.ARENA_CM}x{config.ARENA_CM} cm)')
    ax1.set_xlabel('x (cm)')
    ax1.set_ylabel('y (cm)')

    # 2. Input channels: d_far, d_near, speed
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(t_s, all_inputs[:, 0], label='d_far', linewidth=0.8)
    ax2.plot(t_s, all_inputs[:, 1], label='d_near', linewidth=0.8)
    ax2.plot(t_s, all_inputs[:, 2], label='speed', linewidth=0.8)
    ax2.set_title('Scalar inputs (batch 0)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Rate (Hz)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. HD heatmap
    ax3 = fig.add_subplot(3, 2, 3)
    hd_data = all_inputs[:, 3:21].T  # [18, T]
    # Subsample for display
    step = max(1, hd_data.shape[1] // 1000)
    ax3.imshow(hd_data[:, ::step], aspect='auto', origin='lower',
               extent=[t_s[0], t_s[-1], 0, 18], cmap='hot')
    ax3.set_title('HD inputs (18 channels, batch 0)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('HD channel')

    # 4. Targets
    ax4 = fig.add_subplot(3, 2, 4)
    wall_names = ['N', 'S', 'E', 'W']
    for j in range(4):
        ax4.plot(t_s, all_targets[:, j], label=f'Border_{wall_names[j]}', linewidth=0.8)
    ax4.set_title('Target rates (batch 0)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Rate (Hz)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Target spatial heatmap
    ax5 = fig.add_subplot(3, 2, 5)
    # Use first N steps matching trajectory
    n_show = min(len(traj['x']), all_targets.shape[0])
    x_show = traj['x'][:n_show]
    y_show = traj['y'][:n_show]
    # Plot N wall target as scatter
    sc = ax5.scatter(x_show, y_show, c=all_targets[:n_show, 0],
                     s=0.5, cmap='YlOrRd', vmin=0, vmax=config.F_MAX_BORDER)
    plt.colorbar(sc, ax=ax5, label='Border_N rate (Hz)')
    ax5.set_xlim(0, config.ARENA_CM)
    ax5.set_ylim(0, config.ARENA_CM)
    ax5.set_aspect('equal')
    ax5.set_title('Border_N target (spatial)')

    # 6. Info
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')
    info_text = (
        f"Dataset: {ds_path}\n"
        f"Batches: {n_batches}\n"
        f"Batch steps: {all_inputs.shape[0]}\n"
        f"Batch duration: {config.BATCH_DURATION} s\n"
        f"DT: {config.DT} ms\n"
        f"Arena: {config.ARENA_CM} cm\n"
        f"Inputs: {config.N_INPUTS} channels\n"
        f"Targets: 4 walls\n"
        f"\nInput ranges:\n"
        f"  d_far: [{all_inputs[:, 0].min():.2f}, {all_inputs[:, 0].max():.2f}] Hz\n"
        f"  d_near: [{all_inputs[:, 1].min():.2f}, {all_inputs[:, 1].max():.2f}] Hz\n"
        f"  speed: [{all_inputs[:, 2].min():.2f}, {all_inputs[:, 2].max():.2f}] Hz\n"
        f"  HD: [{all_inputs[:, 3:].min():.2f}, {all_inputs[:, 3:].max():.2f}] Hz\n"
        f"\nTarget ranges:\n"
        f"  [{all_targets.min():.2f}, {all_targets.max():.2f}] Hz"
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
    args = parser.parse_args()

    visualize_dataset(args.dataset, args.trajectory, args.output_dir)


if __name__ == '__main__':
    main()
