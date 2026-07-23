"""Generate dataset: precompute inputs and targets from trajectory.

Usage:
    python generate_dataset.py [--trajectory data/trajectory.h5] [--output data/dataset.h5]
"""

import os
import argparse
import numpy as np
import config
from utils.dataset import (
    load_trajectory_from_hdf5, compute_targets,
    prepare_batches, save_dataset_hdf5,
)
from utils.inputs import precompute_inputs


def generate_dataset(trajectory_path: str = None, output_path: str = None):
    """Load trajectory, compute inputs/targets, save batched dataset."""

    traj_path = trajectory_path or config.TRAJECTORY_HDF5
    out_path = output_path or os.path.join(
        os.path.dirname(config.TRAJECTORY_HDF5), 'dataset.h5')

    print(f"Loading trajectory from {traj_path}...")
    traj = load_trajectory_from_hdf5(traj_path)
    n_steps = traj['x'].shape[1]
    print(f"  Steps: {n_steps}, duration: {traj['t'][0, -1]:.1f} s")

    print("Precomputing inputs (21 channels)...")
    inputs = precompute_inputs(traj)
    print(f"  inputs shape: {inputs.shape}")
    print(f"  d_far range: [{inputs[..., 0].min():.2f}, {inputs[..., 0].max():.2f}] Hz")
    print(f"  d_near range: [{inputs[..., 1].min():.2f}, {inputs[..., 1].max():.2f}] Hz")
    print(f"  speed range: [{inputs[..., 2].min():.2f}, {inputs[..., 2].max():.2f}] Hz")
    print(f"  CB range: [{inputs[..., 3:3 + 8].min():.2f}, {inputs[..., 3:3 + 8].max():.2f}] Hz")
    print(f"  CD×HD range: [{inputs[..., 11:19].min():.2f}, {inputs[..., 11:19].max():.2f}] Hz")
    print(f"  cd_far range: [{inputs[..., 19].min():.2f}, {inputs[..., 19].max():.2f}] Hz")
    print(f"  cd_near range: [{inputs[..., 20].min():.2f}, {inputs[..., 20].max():.2f}] Hz")

    print("Precomputing targets (4 walls)...")
    targets = compute_targets(traj)
    print(f"  targets shape: {targets.shape}")
    print(f"  target range: [{targets.min():.2f}, {targets.max():.2f}] Hz")


    metadata = {
        'trajectory_path': traj_path,
        'n_steps_total': n_steps,
        'batch_duration': config.BATCH_DURATION,
    }
    print(f"Saving dataset to {out_path}...")
    save_dataset_hdf5(out_path, inputs, targets, metadata)
    print(f"Done. Dataset saved to {out_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate dataset from trajectory")
    parser.add_argument("--trajectory", type=str, default=None,
                        help="Path to trajectory HDF5 file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output dataset HDF5 file")
    args = parser.parse_args()

    generate_dataset(args.trajectory, args.output)


if __name__ == '__main__':
    main()
