"""Spatial rate maps of all 21 input channels.

For each batch (0.1s of trajectory), build a 2D spatial rate map per input
channel. Then average across all batches (occupancy-weighted) to get the
long-term spatial distribution of each input.

Equivalent to a single global rate map, but the per-batch-then-average
approach makes the per-channel statistics less sensitive to trajectory
length bias (long stretches in one region of the arena don't dominate
the global average).

Usage:
    python visualize_input_maps.py [--dataset data/dataset.h5]
                                   [--trajectory data/trajectory.h5]
                                   [--bin-size 2.5] [--batch-duration 0.1]
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import config
from utils.dataset import load_trajectory_from_hdf5, load_dataset_hdf5


# Channel layout (must match utils/inputs.py::precompute_inputs)
SCALAR_CHANNELS = [0, 1, 2, 19, 20]    # d_far, d_near, speed, cd_far, cd_near
CB_CHANNELS = list(range(3, 11))        # CB_0..CB_7
CDXHD_CHANNELS = list(range(11, 19))    # CDxHD_0..CDxHD_7
CHANNEL_LABELS = (
    ['d_far', 'd_near', 'speed']
    + [f'CB_{i}' for i in range(config.N_CB)]
    + [f'CDxHD_{i}' for i in range(config.N_HD)]
    + ['cd_far', 'cd_near']
)
assert len(CHANNEL_LABELS) == config.N_INPUTS == 21


def compute_per_batch_rate_maps(traj, X, batch_duration=0.1, bin_size=2.5):
    """Build a spatial rate map for each (batch, channel).

    For each batch (default 0.1s) and each of the 21 input channels,
    compute the mean channel rate at every spatial bin (over the timesteps
    when the agent was inside that bin during this batch). Bins never
    visited in a batch have rate 0.

    Args:
        traj: trajectory dict with x, y (2D arrays, shape (n_trials, n_steps)).
        X: (n_trials, n_steps, 21) input rates in Hz.
        batch_duration: seconds per batch (default 0.1).
        bin_size: cm per spatial bin (default 2.5).

    Returns:
        rate_maps: (n_batches, 21, n_bins, n_bins) — per-batch mean rate per bin.
        occupancies: (n_batches, n_bins, n_bins) — visit count per bin per batch.
    """
    dt_ms = config.DT
    batch_steps = int(batch_duration / (dt_ms / 1000.0))
    n_trials, n_steps, n_inputs = X.shape
    n_batches = n_steps // batch_steps
    n_bins = int(round(config.ARENA_CM / bin_size))

    x = traj['x']   # (n_trials, n_steps)
    y = traj['y']

    rate_maps = np.zeros((n_batches, n_inputs, n_bins, n_bins), dtype=np.float64)
    occupancies = np.zeros((n_batches, n_bins, n_bins), dtype=np.float64)

    for b in range(n_batches):
        start = b * batch_steps
        end = start + batch_steps
        # Flatten all trials × steps in this batch
        xb = np.clip((x[:, start:end] / bin_size).astype(int),
                     0, n_bins - 1).flatten()
        yb = np.clip((y[:, start:end] / bin_size).astype(int),
                     0, n_bins - 1).flatten()
        Xb = X[:, start:end, :].reshape(-1, n_inputs)  # (n_trials*batch_steps, 21)

        for k in range(n_inputs):
            np.add.at(rate_maps[b, k], (yb, xb), Xb[:, k])
        np.add.at(occupancies[b], (yb, xb), 1.0)

    # Normalize per batch (mean rate per bin); unvisited bins stay at 0
    occ_safe = np.maximum(occupancies, 1.0)
    rate_maps = rate_maps / occ_safe[:, np.newaxis, :, :]
    return rate_maps, occupancies


def average_rate_maps(per_batch_maps, per_batch_occ, min_visit_threshold=1):
    """Average per-batch rate maps across all batches, occupancy-weighted.

    For each (channel, bin):
        r_avg = sum_b (r_b * occ_b * [occ_b >= threshold])
              / sum_b (occ_b * [occ_b >= threshold])

    Bins with no visits in any batch yield 0.

    Args:
        per_batch_maps: (n_batches, 21, n_bins, n_bins).
        per_batch_occ:   (n_batches, n_bins, n_bins).
        min_visit_threshold: ignore a batch's contribution for bins it
            visited fewer than this many times (default 1).

    Returns:
        avg_maps: (21, n_bins, n_bins).
        total_occ: (n_bins, n_bins) — total visits per bin.
    """
    n_batches, n_inputs, n_bins, _ = per_batch_maps.shape
    numerator = np.zeros((n_inputs, n_bins, n_bins), dtype=np.float64)
    denominator = np.zeros((n_bins, n_bins), dtype=np.float64)

    for b in range(n_batches):
        mask = per_batch_occ[b] >= min_visit_threshold
        weighted = per_batch_maps[b] * per_batch_occ[b, np.newaxis, :, :]
        numerator += weighted * mask[np.newaxis, :, :]
        denominator += per_batch_occ[b] * mask

    total_occ = denominator.copy()
    avg_maps = numerator / np.maximum(denominator, 1e-12)[np.newaxis, :, :]
    return avg_maps, total_occ


def visualize_input_maps(dataset_path=None, trajectory_path=None,
                         output_dir=None, bin_size=2.5,
                         batch_duration=None):
    """Build and plot per-channel spatial rate maps.

    Layout: 3 rows × 8 cols.
        Row 0: d_far, d_near, speed, cd_far, cd_near, [coverage map],
               [mean-rate bar], [max-rate bar]
        Row 1: CB_0..CB_7
        Row 2: CDxHD_0..CDxHD_7
    """
    ds_path = dataset_path or os.path.join(
        os.path.dirname(config.TRAJECTORY_HDF5), 'dataset.h5')
    traj_path = trajectory_path or config.TRAJECTORY_HDF5
    batch_duration = batch_duration or config.BATCH_DURATION
    out_dir = output_dir or config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading trajectory from {traj_path}...")
    traj = load_trajectory_from_hdf5(traj_path)
    print(f"Loading dataset from {ds_path}...")
    ds = load_dataset_hdf5(ds_path)
    X = ds['X']
    Y = ds['Y']

    if X.ndim != 3 or X.shape[-1] != 21:
        raise ValueError(f"Expected X shape (n_trials, n_steps, 21), got {X.shape}")
    n_trials, n_steps, n_inputs = X.shape

    dt_ms = config.DT
    batch_steps = int(batch_duration / (dt_ms / 1000.0))
    n_batches = n_steps // batch_steps
    n_bins = int(round(config.ARENA_CM / bin_size))

    print(f"Trials: {n_trials}, Steps/trial: {n_steps}, "
          f"Batches/trial: {n_batches}, Total batches: {n_batches * n_trials}")
    print(f"Batch: {batch_duration}s = {batch_steps} steps; "
          f"Grid: {n_bins}×{n_bins} bins of {bin_size} cm")

    print("Computing per-batch rate maps...")
    per_batch_maps, per_batch_occ = compute_per_batch_rate_maps(
        traj, X, batch_duration=batch_duration, bin_size=bin_size)
    # Memory: 10000 batches × 21 channels × 20×20 × 8 bytes ≈ 67 MB
    print(f"  per_batch_maps: {per_batch_maps.shape} "
          f"({per_batch_maps.nbytes / 1e6:.1f} MB)")

    print("Averaging across batches (occupancy-weighted)...")
    avg_maps, total_occ = average_rate_maps(per_batch_maps, per_batch_occ)
    coverage = (total_occ > 0).mean()

    mean_rates = avg_maps.mean(axis=(1, 2))
    max_rates = avg_maps.max(axis=(1, 2))
    vmax_per_channel = np.percentile(avg_maps, 95, axis=(1, 2))
    vmax_per_channel = np.maximum(vmax_per_channel, 1e-6)  # avoid 0

    # Estimate per-channel contribution to the synaptic current I_syn.
    # I_syn_k = g_base * A * rate_k. With the same g_base for all input
    # channels and A roughly constant, the MEAN I_syn contribution is
    # proportional to the channel's mean rate. The total fraction of
    # I_syn from each channel reveals input-scale imbalance.
    total_mean = mean_rates.sum()
    fractions = mean_rates / total_mean

    print(f"  coverage: {coverage * 100:.1f}% of bins visited")
    print(f"  per-channel mean rate (Hz) and fraction of total I_syn:")
    for k, label in enumerate(CHANNEL_LABELS):
        print(f"    {label:8s}  mean={mean_rates[k]:6.3f}  max={max_rates[k]:6.3f}  "
              f"vmax95={vmax_per_channel[k]:6.3f}  "
              f"I_syn_frac={fractions[k]*100:5.1f}%")
    print(f"  Total I_syn share of new egocentric channels "
          f"(CB×8 + CDxHD×8 + cd_far + cd_near): "
          f"{(fractions[3:19].sum() + fractions[19] + fractions[20]) * 100:.1f}%")

    # ----- Figure: 3 rows × 8 cols -----
    fig = plt.figure(figsize=(24, 10))
    gs = GridSpec(3, 8, figure=fig, hspace=0.40, wspace=0.30)

    arena_extent = [0, config.ARENA_CM, 0, config.ARENA_CM]

    def _plot_rate_map(ax, rmap, title, vmax):
        im = ax.imshow(rmap.T, origin='lower', extent=arena_extent,
                       cmap='viridis', vmin=0, vmax=vmax, aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- Row 0: 5 scalar channels + 3 stats panels ----
    for col, k in enumerate(SCALAR_CHANNELS):
        ax = fig.add_subplot(gs[0, col])
        _plot_rate_map(ax, avg_maps[k], CHANNEL_LABELS[k], vmax_per_channel[k])

    # Coverage map
    ax_cov = fig.add_subplot(gs[0, 5])
    im = ax_cov.imshow(total_occ.T, origin='lower', extent=arena_extent,
                      cmap='hot', aspect='equal')
    ax_cov.set_title(f'Coverage ({coverage*100:.1f}%)', fontsize=10)
    ax_cov.set_xticks([])
    ax_cov.set_yticks([])
    plt.colorbar(im, ax=ax_cov, fraction=0.046, pad=0.04, label='visits')

    # Per-channel mean rate
    ax_mean = fig.add_subplot(gs[0, 6])
    ax_mean.barh(np.arange(n_inputs), mean_rates, color='C0')
    ax_mean.set_yticks(np.arange(n_inputs))
    ax_mean.set_yticklabels(CHANNEL_LABELS, fontsize=7)
    ax_mean.invert_yaxis()
    ax_mean.set_xlabel('Mean (Hz)')
    ax_mean.set_title('Mean rate per channel', fontsize=10)
    ax_mean.grid(True, alpha=0.3, axis='x')

    # Per-channel I_syn contribution (proportional to mean rate, since g_base
    # is the same for all inputs). Highlights the input-scale imbalance.
    ax_isyn = fig.add_subplot(gs[0, 7])
    colors = ['C0' if not lbl.startswith(('CB_', 'CDxHD_', 'cd_')) else 'C2'
              for lbl in CHANNEL_LABELS]
    ax_isyn.barh(np.arange(n_inputs), fractions, color=colors)
    ax_isyn.set_yticks(np.arange(n_inputs))
    ax_isyn.set_yticklabels(CHANNEL_LABELS, fontsize=7)
    ax_isyn.invert_yaxis()
    ax_isyn.set_xlabel('Fraction of mean I_syn')
    ax_isyn.set_title('I_syn contribution\n(blue=old, orange=new ego)', fontsize=9)
    ax_isyn.grid(True, alpha=0.3, axis='x')

    # ---- Row 1: CB cells (8 columns) ----
    for col, k in enumerate(CB_CHANNELS):
        ax = fig.add_subplot(gs[1, col])
        pref_deg = config.THETA_PREF_CB[col]
        title = f'CB_{col} (θ_pref={pref_deg:.0f}°)'
        _plot_rate_map(ax, avg_maps[k], title, vmax_per_channel[k])

    # ---- Row 2: CD×HD cells (8 columns) ----
    for col, k in enumerate(CDXHD_CHANNELS):
        ax = fig.add_subplot(gs[2, col])
        pref_deg = config.THETA_PREF_HD[col]
        title = f'CDxHD_{col} (θ_pref={pref_deg:.0f}°)'
        _plot_rate_map(ax, avg_maps[k], title, vmax_per_channel[k])

    plt.suptitle(
        f'Input channel spatial rate maps '
        f'(averaged over {n_batches} batches × {n_trials} trials, '
        f'{bin_size} cm bins, 0.1 s batches)',
        fontsize=13, y=0.995)
    out_path = os.path.join(out_dir, 'input_rate_maps.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return avg_maps, total_occ, mean_rates, max_rates


def main():
    parser = argparse.ArgumentParser(
        description="Spatial rate maps of all 21 input channels")
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--trajectory', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--bin-size', type=float, default=2.5,
                        help='Spatial bin size in cm (default 2.5, '
                             'matches article)')
    parser.add_argument('--batch-duration', type=float,
                        default=config.BATCH_DURATION,
                        help='Batch duration in seconds '
                             f'(default {config.BATCH_DURATION})')
    args = parser.parse_args()
    visualize_input_maps(args.dataset, args.trajectory, args.output_dir,
                         args.bin_size, args.batch_duration)


if __name__ == '__main__':
    main()
