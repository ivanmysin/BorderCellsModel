"""Dataset preparation: convert trajectories to training batches."""

import numpy as np
import h5py
import config
from utils.trajectory import TrajectoryGenerator, generate_trajectory_batch, interpolate_trajectory


def trajectory_to_extra_inputs(traj: dict) -> np.ndarray:
    """Convert trajectory dict to extra_inputs array.

    extra_inputs columns:
        0: x (cm)
        1: y (cm)
        2: vx = speed * cos(head_direction) (cm/s)
        3: vy = speed * sin(head_direction) (cm/s)
    """
    speed = traj["speed"]
    hd = traj["head_direction"]
    extra = np.stack([
        traj["x"], traj["y"],
        speed * np.cos(hd),
        speed * np.sin(hd),
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

    Returns a dict with keys: x, y, vx, vy, speed, head_direction,
    d_N, d_S, d_E, d_W, d_min, t — each shape (n_steps,).
    """
    p = path or config.TRAJECTORY_HDF5
    traj = {}
    with h5py.File(p, "r") as f:
        for key in f.keys():
            traj[key] = f[key][:]

    if 't' not in traj:
        total_steps = len(traj.get('vx', traj.get('x', [])))
        traj['t'] = np.arange(total_steps, dtype=np.float32) * config.TRAJECTORY_DT

    if 'vx' not in traj and 'vy' not in traj and 'speed' in traj and traj['speed'].ndim == 2:
        vx = traj['speed'][:, 0]
        vy = traj['speed'][:, 1]
        traj['head_direction'] = np.arctan2(vy, vx)
        traj['speed'] = np.sqrt(vx**2 + vy**2)
        traj['vx'] = vx
        traj['vy'] = vy

    if slice_duration is not None and slice_duration < traj['t'][-1]:
        dt_actual = traj['t'][1] - traj['t'][0] if len(traj['t']) > 1 else 1.0
        slice_steps = int(slice_duration / dt_actual)
        total = len(traj['t'])
        if slice_steps < total:
            start = np.random.randint(0, total - slice_steps)
            for key in traj:
                traj[key] = traj[key][start:start + slice_steps]

    return traj


def prepare_batch(gen: TrajectoryGenerator = None,
                  duration: float = None,
                  batch_size: int = 1,
                  start_step: int = None,
                  n_steps: int = None) -> dict:
    """Generate trajectory and prepare training batch.

    If 'gen' is None, loads from HDF5.
    If 'duration' is None, uses config.TRIAL_DURATION.
    If start_step and n_steps given, slice the trajectory (HDF5 only).

    Returns:
        t_seq:      [batch, n_steps_neural, 1] time steps (ms)
        extra_seq:  [batch, n_steps_neural, 4] extra inputs [x, y, vx, vy]
        targets:    [batch, n_steps_neural, 4] target border rates
        traj:       dict of raw trajectory arrays
    """
    up = config.UP_SAMPLE_FACTOR
    dur = duration if duration is not None else config.TRIAL_DURATION

    target_dt = config.DT / 1000.0
    if gen is not None:
        raw = generate_trajectory_batch(gen, dur, n_trajectories=batch_size)
        interped = []
        for i in range(batch_size):
            single = {k: raw[k][i] for k in raw}
            interped.append(interpolate_trajectory(single, target_dt))
        traj = {k: np.stack([t[k] for t in interped], axis=0) for k in interped[0]}
    else:
        raw_traj = load_trajectory_from_hdf5()
        if start_step is not None and n_steps is not None:
            end_step = min(start_step + n_steps, len(raw_traj['x']))
            raw_traj = {k: v[start_step:end_step] for k, v in raw_traj.items()}
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
