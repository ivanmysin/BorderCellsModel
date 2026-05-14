"""Dataset preparation: convert trajectories to training batches."""

import numpy as np
import h5py
import config
from utils.trajectory import TrajectoryGenerator, generate_trajectory_batch


def trajectory_to_extra_inputs(traj: dict) -> np.ndarray:
    """Convert trajectory dict to extra_inputs array.

    extra_inputs columns:
        0: d_min (cm)
        1: speed (cm/s)
        2: cos(head_direction)
        3: sin(head_direction)
        4: d_N (cm)
        5: d_S (cm)
        6: d_E (cm)
        7: d_W (cm)
    """
    hd = traj["head_direction"]
    extra = np.stack([
        traj["d_min"],
        traj["speed"],
        np.cos(hd),
        np.sin(hd),
        traj["d_N"],
        traj["d_S"],
        traj["d_E"],
        traj["d_W"],
    ], axis=-1)
    return extra


def compute_targets(traj: dict) -> np.ndarray:
    """Compute target firing rates for 4 border populations.

    r̂_w(t) = f_max_border * exp(-d_w(t) / lambda)
    Returns: shape (..., 4), columns: [r̂_N, r̂_S, r̂_E, r̂_W]
    """
    lbd = config.LAMBDA_PROX
    fmax = config.F_MAX_BORDER
    targets = fmax * np.exp(-np.stack([
        traj["d_N"], traj["d_S"], traj["d_E"], traj["d_W"]
    ], axis=-1) / lbd)
    return targets


def upsample(data: np.ndarray, factor: int) -> np.ndarray:
    """Repeat each step 'factor' times along the time axis."""
    if data.ndim >= 2:
        return np.repeat(data, factor, axis=1)
    return np.repeat(data, factor, axis=0)


def make_t_sequence(n_steps: int) -> np.ndarray:
    """Create time sequence tensor [1, n_steps, 1] in ms."""
    t = np.arange(n_steps, dtype=np.float32) * config.SIM_DT
    return t.reshape(1, -1, 1)


def load_trajectory_from_hdf5(path: str = None, slice_duration: float = None) -> dict:
    """Load trajectory from HDF5, optionally taking a random slice.

    Returns a dict with keys: x, y, speed, head_direction,
    d_N, d_S, d_E, d_W, d_min, t — each shape (n_steps,).
    """
    p = path or config.TRAJECTORY_HDF5
    traj = {}
    with h5py.File(p, "r") as f:
        for key in f.keys():
            traj[key] = f[key][:]

    if slice_duration is not None and slice_duration < traj["t"][-1]:
        dt = config.TRAJECTORY_DT
        slice_steps = int(slice_duration / dt)
        total_steps = len(traj["t"])
        if slice_steps < total_steps:
            start = np.random.randint(0, total_steps - slice_steps)
            for key in traj:
                traj[key] = traj[key][start:start + slice_steps]

    return traj


def prepare_batch(gen: TrajectoryGenerator = None,
                  duration: float = None,
                  batch_size: int = 1) -> dict:
    """Generate trajectory and prepare training batch.

    If 'gen' is None, loads from HDF5.
    If 'duration' is None, uses config.TRIAL_DURATION.

    Returns:
        t_seq:      [batch, n_steps_neural, 1] time steps (ms)
        extra_seq:  [batch, n_steps_neural, 8] extra inputs
        targets:    [batch, n_steps_neural, 4] target border rates
        traj:       dict of raw trajectory arrays
    """
    up = config.UP_SAMPLE_FACTOR
    dur = duration if duration is not None else config.TRIAL_DURATION

    if gen is not None:
        traj = generate_trajectory_batch(gen, dur, n_trajectories=batch_size)
    else:
        raw_traj = load_trajectory_from_hdf5(slice_duration=dur)
        traj = {k: v[np.newaxis, ...] for k, v in raw_traj.items()}

    extra = trajectory_to_extra_inputs(traj)
    targets = compute_targets(traj)

    if up > 1:
        extra = upsample(extra, up)
        targets = upsample(targets, up)

    n_steps = extra.shape[1]
    t_seq = make_t_sequence(n_steps)

    return {
        "t_seq": t_seq.astype(np.float32),
        "extra_seq": extra.astype(np.float32),
        "targets": targets.astype(np.float32),
        "traj": traj,
    }
