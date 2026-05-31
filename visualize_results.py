"""Visualize training results: loss curves, rate maps, tuning curves.

Usage:
    python visualize_results.py [--training results/training.h5]
                                 [--dataset data/dataset.h5]
                                 [--trajectory data/trajectory.h5]
"""

import os
import argparse
import numpy as np
import h5py
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
from train import build_network, graph_pack_inputs
from utils.dataset import load_dataset_hdf5, load_trajectory_from_hdf5


def visualize_results(training_path: str = None, dataset_path: str = None,
                      trajectory_path: str = None, output_dir: str = None):

    train_path = training_path or os.path.join(config.RESULTS_DIR, 'training.h5')
    ds_path = dataset_path or os.path.join(
        os.path.dirname(config.TRAJECTORY_HDF5), 'dataset.h5')
    traj_path = trajectory_path or config.TRAJECTORY_HDF5
    out_dir = output_dir or config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Load training results
    print(f"Loading training results from {train_path}...")
    with h5py.File(train_path, 'r') as f:
        loss_history = f['loss_history'][:]

    # Load trajectory for spatial plots
    print(f"Loading trajectory from {traj_path}...")
    traj = load_trajectory_from_hdf5(traj_path)

    # Load dataset for a batch
    print(f"Loading dataset from {ds_path}...")
    ds = load_dataset_hdf5(ds_path)
    batch0 = ds['get_batch'](0)

    # Run network forward on batch 0
    print("Running forward simulation...")
    network = build_network()
    # Load trained weights
    with h5py.File(train_path, 'r') as f:
        param_grp = f['parameters']
        for v in network.trainable_variables:
            name = v.name.replace(':', '_').replace('/', '_')
            if name in param_grp:
                v.assign(param_grp[name][:])

    t_seq = tf.constant(batch0['t_seq'])
    inputs = graph_pack_inputs(network, batch0['inputs'])
    output = network(t_seq, inputs=inputs, training=False)
    pred_rates = output.firing_rates[config.POPULATION_NAME].numpy()[0]  # [T, 6]
    targets = batch0['targets'][0]  # [T, 4]
    ds['file'].close()

    # --- 1. Loss curve ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history, linewidth=1.5, color='tab:blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training Loss (final={loss_history[-1]:.6f})')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, 'loss_curve.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved loss_curve.png")

    # --- 2. Predicted vs Target ---
    wall_names = ['N', 'S', 'E', 'W']
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    t_s = np.arange(pred_rates.shape[0]) * config.DT / 1000.0
    for j, (ax, wall) in enumerate(zip(axes.flat, wall_names)):
        ax.plot(t_s, targets[:, j], label='Target', linewidth=1.5,
                color='tab:green', alpha=0.8)
        ax.plot(t_s, pred_rates[:, j], label='Predicted', linewidth=1.0,
                color='tab:red', linestyle='--', alpha=0.8)
        ax.set_title(f'Border_{wall}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rate (Hz)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Predicted vs Target (batch 0)', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, 'pred_vs_target.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved pred_vs_target.png")

    # --- 3. Rate maps (spatial) ---
    n_show = min(len(traj['x']), pred_rates.shape[0])
    x_show = traj['x'][:n_show]
    y_show = traj['y'][:n_show]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for j, (ax, wall) in enumerate(zip(axes.flat, wall_names)):
        sc = ax.scatter(x_show, y_show, c=pred_rates[:n_show, j],
                        s=0.5, cmap='YlOrRd', vmin=0,
                        vmax=max(pred_rates[:, j].max(), 1.0))
        plt.colorbar(sc, ax=ax, label='Rate (Hz)')
        ax.set_xlim(0, config.ARENA_CM)
        ax.set_ylim(0, config.ARENA_CM)
        ax.set_aspect('equal')
        ax.set_title(f'Border_{wall} predicted rate map')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
    plt.suptitle('Predicted Rate Maps', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, 'rate_maps.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved rate_maps.png")

    # --- 4. Basket / Axo activity ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for j, (ax, name) in enumerate(zip(axes, ['Basket', 'Axo'])):
        ax.plot(t_s, pred_rates[:, 4 + j], linewidth=0.8, color=f'C{j+4}')
        ax.set_title(f'{name} predicted activity')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Rate (Hz)')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'inhibitory_activity.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved inhibitory_activity.png")

    # --- 5. Trained parameter matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    with h5py.File(train_path, 'r') as f:
        param_grp = f['parameters']
        for v in network.trainable_variables:
            name = v.name.replace(':', '_').replace('/', '_')
            arr = v.numpy()
            if 'gsyn_max' in name and arr.ndim == 2:
                if arr.shape == (config.N_INPUTS, config.N_POP_UNITS):
                    ax = axes[0]
                    ax.set_title('Input gsyn_max [21x6]')
                    im = ax.imshow(arr, aspect='auto', cmap='viridis')
                    plt.colorbar(im, ax=ax)
                    ax.set_xlabel('Population unit')
                    ax.set_ylabel('Input channel')
                elif arr.shape == (config.N_POP_UNITS, config.N_POP_UNITS):
                    ax = axes[1]
                    ax.set_title('Recurrent gsyn_max [6x6]')
                    im = ax.imshow(arr, cmap='viridis')
                    plt.colorbar(im, ax=ax)
                    ax.set_xticks(range(6))
                    ax.set_yticks(range(6))
                    ax.set_xticklabels(config.UNIT_NAMES, fontsize=7, rotation=45)
                    ax.set_yticklabels(config.UNIT_NAMES, fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'trained_weights.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved trained_weights.png")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("--training", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--trajectory", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    visualize_results(args.training, args.dataset, args.trajectory, args.output_dir)


if __name__ == '__main__':
    main()
