"""Visualize training results and dynamics from dynamics.h5.

Reads pre-computed firing rates from results/dynamics.h5 (produced by
simulate_dynamics.py) and produces spatial rate maps for all 6 neurons,
plus loss curve, pred-vs-target and trained weights.

Usage:
    python visualize_results.py [--training results/training.h5]
                                 [--dynamics results/dynamics.h5]
                                 [--trajectory data/trajectory.h5]
                                 [--output-dir results]
"""

import os
import argparse

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import config
from utils.dataset import load_trajectory_from_hdf5
from utils.trajectory import interpolate_trajectory


def binned_rate_map(x, y, rate, n_bins=50, smooth_sigma=1.5):
    """2D binned rate map: mean(rate) in each (x, y) cell, with Gaussian smoothing."""
    arena = config.ARENA_CM
    edges = np.linspace(0, arena, n_bins + 1)
    rate_sum, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=rate)
    occupancy, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_map = np.where(occupancy > 0, rate_sum / occupancy, 0.0)
    rate_map_s = gaussian_filter(rate_map, sigma=smooth_sigma)
    return rate_map_s, [0, arena, 0, arena]


def visualize_results(training_path=None, dynamics_path=None,
                      trajectory_path=None, output_dir=None):
    train_path = training_path or os.path.join(config.RESULTS_DIR, 'training.h5')
    dyn_path = dynamics_path or os.path.join(config.RESULTS_DIR, 'dynamics.h5')
    traj_path = trajectory_path or config.TRAJECTORY_HDF5
    out_dir = output_dir or config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading training history from {train_path}...")
    with h5py.File(train_path, 'r') as f:
        loss_history = f['loss_history'][:]
        trained_params = {k: v[:] for k, v in f['parameters'].items()}

    print(f"Loading dynamics from {dyn_path}...")
    with h5py.File(dyn_path, 'r') as f:
        rates = f['rates'][:]
        targets = f['targets'][:]
        start_batch = int(f.attrs.get('start_batch', 0))
        end_batch = int(f.attrs.get('end_batch', rates.shape[0]))
    print(f"  rates shape: {rates.shape}, targets shape: {targets.shape}")

    rates_flat = rates.reshape(-1, rates.shape[-1])
    targets_flat = targets.reshape(-1, targets.shape[-1])
    n_total_steps = rates_flat.shape[0]
    T = rates.shape[2]

    print(f"Loading trajectory from {traj_path}...")
    traj = load_trajectory_from_hdf5(traj_path)
    print(f"  Interpolating to neural dt = {config.DT} ms ...")
    traj_fine = interpolate_trajectory(traj, config.DT / 1000.0)

    offset = start_batch * T
    if offset + n_total_steps <= len(traj_fine['x']):
        x_all = traj_fine['x'][offset:offset + n_total_steps]
        y_all = traj_fine['y'][offset:offset + n_total_steps]
    else:
        available = len(traj_fine['x']) - offset
        print(f"WARNING: trajectory shorter than needed "
              f"({available} < {n_total_steps}), truncating dynamics")
        n_total_steps = available
        rates_flat = rates_flat[:n_total_steps]
        targets_flat = targets_flat[:n_total_steps]
        x_all = traj_fine['x'][offset:offset + n_total_steps]
        y_all = traj_fine['y'][offset:offset + n_total_steps]
    print(f"  Total steps: {n_total_steps}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history, linewidth=1.5, color='tab:blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSLE)')
    ax.set_title(f'Training Loss (final={loss_history[-1]:.6f}, '
                 f'best={min(loss_history):.6f})')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, 'loss_curve.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved loss_curve.png")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for j, (ax, name) in enumerate(zip(axes.flat, config.UNIT_NAMES)):
        rm, extent = binned_rate_map(x_all, y_all, rates_flat[:, j],
                                     n_bins=50, smooth_sigma=1.5)
        vmax = max(rm.max(), 1e-3)
        im = ax.imshow(rm.T, origin='lower', extent=extent,
                       cmap='hot', vmin=0, vmax=vmax, aspect='equal')
        plt.colorbar(im, ax=ax, label='Rate (Hz)')
        ax.set_title(f'{name} (mean={rates_flat[:, j].mean():.2f}, '
                     f'max={rates_flat[:, j].max():.2f} Hz)')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
    plt.suptitle(f'Spatial Rate Maps (binned, smoothed) — '
                 f'batches [{start_batch}, {end_batch})', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, 'rate_maps_all.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved rate_maps_all.png")

    step = max(1, n_total_steps // 50000)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for j, (ax, name) in enumerate(zip(axes.flat, config.UNIT_NAMES)):
        sc = ax.scatter(x_all[::step], y_all[::step],
                        c=rates_flat[::step, j], s=0.3, cmap='YlOrRd',
                        vmin=0, vmax=max(rates_flat[:, j].max(), 1e-3))
        plt.colorbar(sc, ax=ax, label='Rate (Hz)')
        ax.set_xlim(0, config.ARENA_CM)
        ax.set_ylim(0, config.ARENA_CM)
        ax.set_aspect('equal')
        ax.set_title(f'{name} (mean={rates_flat[:, j].mean():.2f} Hz)')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
    plt.suptitle('Spatial Rate Maps (raw scatter, downsampled)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, 'rate_maps_scatter.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved rate_maps_scatter.png")

    wall_names = ['N', 'S', 'E', 'W']
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    t_first = np.arange(rates.shape[1]) * config.DT / 1000.0



    for j, (ax, wall) in enumerate(zip(axes.flat, wall_names)):
        ax.plot(t_first, targets[3, :, j], label='Target',
                linewidth=1.5, color='tab:green', alpha=0.8)
        ax.plot(t_first, rates[3, :, j], label='Predicted',
                linewidth=1.0, color='tab:red', linestyle='--', alpha=0.8)
        ax.set_title(f'Border_{wall} (batch {start_batch})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rate (Hz)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'Predicted vs Target (first batch, t=0..{t_first[-1]:.1f}s)',
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, 'pred_vs_target.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("  Saved pred_vs_target.png")

    gsyn = trained_params.get('gsyn_max_0',
             trained_params.get('gsyn_max', None))
    if gsyn is not None and gsyn.ndim == 2:
        n_pop = config.N_POP_UNITS
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        im = axes[0].imshow(gsyn[n_pop:], aspect='auto', cmap='viridis')
        axes[0].set_title('Input gsyn_max [21x6]')
        axes[0].set_xlabel('Population unit')
        axes[0].set_ylabel('Input channel')
        plt.colorbar(im, ax=axes[0])
        im = axes[1].imshow(gsyn[:n_pop], cmap='viridis')
        axes[1].set_title('Recurrent gsyn_max [6x6]')
        axes[1].set_xticks(range(n_pop))
        axes[1].set_yticks(range(n_pop))
        axes[1].set_xticklabels(config.UNIT_NAMES, fontsize=7, rotation=45)
        axes[1].set_yticklabels(config.UNIT_NAMES, fontsize=7)
        plt.colorbar(im, ax=axes[1])
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, 'trained_weights.png'),
                    bbox_inches='tight', dpi=150)
        plt.close(fig)
        print("  Saved trained_weights.png")
    else:
        print("  (skipped trained_weights.png: gsyn_max not found)")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=str, default=None)
    parser.add_argument("--dynamics", type=str, default=None)
    parser.add_argument("--trajectory", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    visualize_results(args.training, args.dynamics,
                      args.trajectory, args.output_dir)


if __name__ == '__main__':
    main()
