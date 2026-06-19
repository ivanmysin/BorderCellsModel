"""Generate minimal test dataset: 4 batches of 1s each.

Agent runs from center toward each wall at 25 cm/s:
  Batch 0 → North  (y increases)
  Batch 1 → South  (y decreases)
  Batch 2 → East   (x increases)
  Batch 3 → West   (x decreases)

Usage:
    python generate_test_dataset.py [--output data/test_dataset.h5]
"""
import os
import numpy as np
import h5py
import config
from utils.inputs import precompute_inputs
from utils.dataset import compute_targets, prepare_batches


def make_trajectory(wall_idx, duration=1.0, speed_cm=25.0):
    """Straight-line trajectory from center toward one wall.

    wall_idx: 0=N, 1=S, 2=E, 3=W
    Returns dict with same keys as trajectory.h5.
    """
    dt_s = config.DT / 1000.0
    n_steps = int(duration / dt_s)
    t = np.arange(n_steps, dtype=np.float32) * dt_s

    cx, cy = config.ARENA_CM / 2.0, config.ARENA_CM / 2.0

    # direction vectors: N(+y), S(-y), E(+x), W(-x)
    dx_map = {0: 0.0, 1: 0.0, 2: speed_cm, 3: -speed_cm}
    dy_map = {0: speed_cm, 1: -speed_cm, 2: 0.0, 3: 0.0}
    dx, dy = dx_map[wall_idx], dy_map[wall_idx]

    x = cx + dx * t
    y = cy + dy * t

    # clip to arena
    x = np.clip(x, 0.1, config.ARENA_CM - 0.1)
    y = np.clip(y, 0.1, config.ARENA_CM - 0.1)

    vx = np.full(n_steps, dx, dtype=np.float32)
    vy = np.full(n_steps, dy, dtype=np.float32)
    speed = np.full(n_steps, speed_cm, dtype=np.float32)
    head_direction = np.full(n_steps, np.arctan2(dy, dx), dtype=np.float32)

    arena = config.ARENA_CM
    d_N = arena - y
    d_S = y
    d_E = arena - x
    d_W = x
    d_min = np.minimum(np.minimum(d_N, d_S), np.minimum(d_E, d_W))

    return {
        'x': x.astype(np.float32),
        'y': y.astype(np.float32),
        'vx': vx, 'vy': vy,
        'speed': speed,
        'head_direction': head_direction,
        'd_N': d_N.astype(np.float32),
        'd_S': d_S.astype(np.float32),
        'd_E': d_E.astype(np.float32),
        'd_W': d_W.astype(np.float32),
        'd_min': d_min.astype(np.float32),
        't': (t * 1000).astype(np.float32),  # ms
    }


def main():
    output_path = os.path.join(
        os.path.dirname(config.TRAJECTORY_HDF5), 'test_dataset.h5')

    wall_names = ['North', 'South', 'East', 'West']
    duration = 1.0
    n_steps = int(duration / (config.DT / 1000.0))

    print(f"Arena: {config.ARENA_CM} cm, DT: {config.DT} ms")
    print(f"Steps per batch: {n_steps}, Duration: {duration} s")
    print(f"Speed: 25 cm/s, 4 batches\n")

    all_batches = []
    for i, name in enumerate(wall_names):
        traj = make_trajectory(i, duration=duration, speed_cm=25.0)
        inputs = precompute_inputs(traj)       # (n_steps, 21)
        targets = compute_targets(traj)         # (n_steps, 4)

        # wrap into prepare_batches format
        t_seq = np.arange(n_steps, dtype=np.float32) * config.DT
        t_seq = t_seq.reshape(1, -1, 1)

        batch = {
            't_seq': t_seq,
            'inputs': inputs[np.newaxis, :, :].astype(np.float32),
            'targets': targets[np.newaxis, :, :].astype(np.float32),
        }
        all_batches.append(batch)

        d_min_range = f"[{traj['d_min'].min():.1f}, {traj['d_min'].max():.1f}]"
        tgt_range = f"[{targets.min():.2f}, {targets.max():.2f}]"
        print(f"  Batch {i} ({name:5s}): "
              f"d_min={d_min_range} cm, target={tgt_range} Hz, "
              f"input ch0(d_far)=[{inputs[:, 0].min():.2f}, {inputs[:, 0].max():.2f}]")

    print(f"\nSaving {len(all_batches)} batches to {output_path}...")
    metadata = {
        'description': 'test_dataset: 4 batches toward each wall at 25 cm/s',
        'batch_duration': duration,
    }

    with h5py.File(output_path, 'w') as f:
        ds_grp = f.create_group('dataset')
        ds_grp.attrs['n_batches'] = len(all_batches)
        ds_grp.attrs['batch_steps'] = n_steps
        ds_grp.attrs['dt'] = config.DT
        ds_grp.attrs['arena_cm'] = config.ARENA_CM
        for k, v in metadata.items():
            ds_grp.attrs[k] = v
        for i, batch in enumerate(all_batches):
            grp = ds_grp.create_group(f'batch_{i}')
            grp.create_dataset('t_seq', data=batch['t_seq'])
            grp.create_dataset('inputs', data=batch['inputs'])
            grp.create_dataset('targets', data=batch['targets'])

    print(f"Done. File: {output_path}")
    print(f"  X shape per batch: (1, {n_steps}, {config.N_INPUTS})")
    print(f"  Y shape per batch: (1, {n_steps}, 4)")


if __name__ == '__main__':
    main()
