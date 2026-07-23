"""Visualize dataset + apply Long et al. (2025) formulas to the precomputed inputs
and the 4 border-cell targets.

Article formulas implemented (Nature Comms, 2025; see also context/allo_equations.md):

  - Center-bearing (CB): CB = atan2(sin(θ_to_C - θ_HD), cos(θ_to_C - θ_HD))
    Verification per CB cell k: MV = sum_t r_k(t)·exp(i·CB(t)) / sum_t r_k(t)
                                MVL = |MV|, preferred_bearing = angle(MV)

  - Center-distance (CD): CD = √((x_C - x)² + (y_C - y)²)
    Verification: linear fit R² of cd_far / cd_near / CD×HD cells vs CD

  - Border score (Lever et al. 2009; cited in the article):
        b = (c_M - d_m) / (c_M + d_m)
    where c_M = maximal extent along a wall of any firing field touching that wall
          d_m = mean distance of all firing-field centroids to the nearest wall
    Fields defined as connected regions of bins with rate > 0.3·max_rate, area ≥ 200 cm².

  - Egocentric boundary cell (EBC) Mean Vector (article Methods):
        MV = sum_{θ,d} F_{θ,d} · exp(i·θ) / sum_{θ,d} F_{θ,d}
        MVL = |MV|
    Built from a 2D polar rate map (bearing × distance) of the cell's firing rate
    weighted by the egocentric bearing+distance to the NEAREST wall at each timestep.

Usage:
    python visualize_dataset.py [--dataset data/dataset.h5] [--trajectory data/trajectory.h5]
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import label as nd_label
import config
from utils.dataset import load_trajectory_from_hdf5, load_dataset_hdf5


# ============================================================
# Article formulas
# ============================================================

def compute_cb(traj: dict) -> np.ndarray:
    """Center-bearing: egocentric bearing to geometric center.

    CB = atan2(sin(θ_to_C - θ_HD), cos(θ_to_C - θ_HD)),  range (-π, π].

    Args:
        traj: dict with keys x, y, head_direction, all 2D (n_trials, n_steps).

    Returns:
        np.ndarray of shape (n_trials, n_steps) in radians.
    """
    x, y, hd = traj['x'], traj['y'], traj['head_direction']
    cx, cy = config.ARENA_CM / 2.0, config.ARENA_CM / 2.0
    theta_to_center = np.arctan2(cy - y, cx - x)
    delta = theta_to_center - hd
    return np.arctan2(np.sin(delta), np.cos(delta))


def compute_cd(traj: dict) -> np.ndarray:
    """Center-distance: distance to geometric center.

    Args:
        traj: dict with keys x, y, all 2D.

    Returns:
        np.ndarray of shape (n_trials, n_steps) in cm.
    """
    x, y = traj['x'], traj['y']
    cx, cy = config.ARENA_CM / 2.0, config.ARENA_CM / 2.0
    return np.sqrt((cx - x) ** 2 + (cy - y) ** 2)


def circular_mvl(signals: np.ndarray, angles: np.ndarray) -> tuple:
    """Compute Mean Vector Length and preferred angle for a population of cells.

    Per-cell formula (article, "Identification of head-direction cells"):
        MV = sum_t F(t)·exp(i·θ(t)) / sum_t F(t)
        MVL = |MV|, preferred = angle(MV)

    Sums over ALL leading dimensions of `signals` (e.g. both trials and steps).

    Args:
        signals: (..., n_cells) firing rates.
        angles: (...) angles in radians, broadcastable with signals[..., 0].

    Returns:
        mvl: (n_cells,) ∈ [0, 1].
        pref_angle: (n_cells,) in radians, ∈ (-π, π].
    """
    # Flatten all leading dims to a single (N, n_cells)
    n_cells = signals.shape[-1]
    sig_flat = signals.reshape(-1, n_cells)
    ang_flat = angles.reshape(-1)
    z = sig_flat * np.exp(1j * ang_flat[:, None])  # (N, n_cells)
    z_sum = z.sum(axis=0)                           # (n_cells,)
    total = sig_flat.sum(axis=0)                    # (n_cells,)
    mvl = np.abs(z_sum) / np.maximum(total, 1e-12)
    pref = np.angle(z_sum)
    return mvl, pref


def linear_fit_R2(signals: np.ndarray, predictor: np.ndarray) -> tuple:
    """R² and slope of linear fit of signals vs predictor (per cell).

    Sums over ALL leading dimensions of `signals`.

    Args:
        signals: (..., n_cells)
        predictor: (...) same leading dims.

    Returns:
        R2: (n_cells,) ∈ [0, 1] (sign of slope dropped).
        slope: (n_cells,)
    """
    n_cells = signals.shape[-1]
    sig_flat = signals.reshape(-1, n_cells)
    pred_flat = predictor.reshape(-1)
    pred_centered = pred_flat - pred_flat.mean()
    sig_centered = sig_flat - sig_flat.mean(axis=0, keepdims=True)
    cov = (sig_centered * pred_centered[:, None]).sum(axis=0)   # (n_cells,)
    var_pred = (pred_centered ** 2).sum()
    var_sig = (sig_centered ** 2).sum(axis=0)
    slope = cov / np.maximum(var_pred, 1e-12)
    R2 = (cov ** 2) / np.maximum(var_pred * var_sig, 1e-12)
    return R2, slope


def compute_spatial_rate_map(traj: dict, signals: np.ndarray, bin_size: float = 1.0) -> np.ndarray:
    """Build 2D spatial rate map of a (single) cell's firing.

    Args:
        traj: dict with x, y (2D, possibly 1 trial if signals is 1D).
        signals: (n_steps,) or (n_trials, n_steps) firing rates.
        bin_size: cm per bin.

    Returns:
        rate_map: (n_bins, n_bins) array, mean firing rate per bin.
    """
    A = config.ARENA_CM
    n_bins = int(round(A / bin_size))
    x = np.atleast_2d(traj['x'])
    y = np.atleast_2d(traj['y'])
    if signals.ndim == 1:
        signals = signals[None, :]
    x_flat = x.flatten()
    y_flat = y.flatten()
    s_flat = signals.flatten()
    xb = np.clip((x_flat / bin_size).astype(int), 0, n_bins - 1)
    yb = np.clip((y_flat / bin_size).astype(int), 0, n_bins - 1)
    rate_sum = np.zeros((n_bins, n_bins), dtype=np.float64)
    occ = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(rate_sum, (yb, xb), s_flat)
    np.add.at(occ, (yb, xb), 1.0)
    return np.where(occ > 0, rate_sum / np.maximum(occ, 1.0), 0.0)


def compute_border_score(rate_map: np.ndarray, bin_size: float = 1.0,
                         min_field_area_cm2: float = 200.0,
                         threshold_frac: float = 0.3,
                         smooth_sigma_bins: float = 2.5) -> dict:
    """Border score per Lever et al. 2009 (cited in the article).

        b = (c_M - d_m) / (c_M + d_m)

    ADAPTED for rat trajectories that don't quite reach the wall: rather than
    requiring the field to LITERALLY touch the wall, we use the field's extent
    along the nearest-wall axis as c_M, and the centroid's distance to the
    nearest wall as d_m. This is the same formula, but softens the touching
    requirement so non-touching (but wall-adjacent) fields still score > 0.

    Also reports a robust 1D variant `b_robust`:
        b_robust = (mean_rate_in_wall_band - mean_rate_in_interior) /
                   (mean_rate_in_wall_band + mean_rate_in_interior)
    with a 5 cm wall band.

    Per the article's Methods ("Spike sorting and behavioral correlates"),
    the rate map is smoothed with a quasi-Gaussian kernel over a 5×5-bin
    neighborhood BEFORE field detection. We apply a Gaussian smooth with
    sigma=smooth_sigma_bins (≈ the article's effective kernel radius / 2)
    to bridge small coverage gaps and avoid fragmentation of fields.

    Fields: connected regions of bins with rate > threshold_frac * max_rate,
    area ≥ min_field_area_cm2.

    Args:
        rate_map: (n_bins, n_bins) 2D spatial rate map.
        bin_size: cm per bin (must match rate_map's resolution).
        min_field_area_cm2: minimum field area (default 200, article value).
        threshold_frac: fraction of peak rate defining a field (default 0.3).
        smooth_sigma_bins: Gaussian sigma (in bins) applied before thresholding.

    Returns:
        dict with 'b', 'c_M', 'd_m', 'n_fields', 'b_robust',
             'field_areas', 'field_d_mins'.
    """
    from scipy.ndimage import gaussian_filter
    A = config.ARENA_CM
    n_bins = rate_map.shape[0]
    smoothed = gaussian_filter(rate_map, sigma=smooth_sigma_bins, mode='constant', cval=0.0)
    max_rate = smoothed.max()
    if max_rate <= 0:
        return _empty_border_result()

    # ---- Field detection (article criterion) ----
    mask = smoothed > threshold_frac * max_rate
    labeled, n_fields = nd_label(mask)
    if n_fields == 0:
        return _empty_border_result()

    field_areas = []
    field_d_mins = []
    c_M_candidates = []

    for f in range(1, n_fields + 1):
        field_mask = (labeled == f)
        ys, xs = np.where(field_mask)
        area_cm2 = field_mask.sum() * bin_size ** 2
        if area_cm2 < min_field_area_cm2:
            continue
        centroid_x_cm = (xs + 0.5) * bin_size
        centroid_y_cm = (ys + 0.5) * bin_size
        cx = centroid_x_cm.mean()
        cy = centroid_y_cm.mean()
        d_n = A - cy
        d_s = cy
        d_e = A - cx
        d_w = cx
        d_min = min(d_n, d_s, d_e, d_w)
        # c_M: extent along the nearest-wall axis
        # (N/S walls run along x, E/W walls run along y)
        if d_min == d_n or d_min == d_s:
            c_M_field = (xs.max() - xs.min() + 1) * bin_size
        else:
            c_M_field = (ys.max() - ys.min() + 1) * bin_size
        field_areas.append(area_cm2)
        field_d_mins.append(d_min)
        c_M_candidates.append(c_M_field)

    # ---- Robust 1D border score (wall band vs interior) ----
    band_width_bins = max(1, int(5.0 / bin_size))  # 5 cm band
    wall_band = np.zeros_like(rate_map, dtype=bool)
    wall_band[:band_width_bins, :] = True
    wall_band[-band_width_bins:, :] = True
    wall_band[:, :band_width_bins] = True
    wall_band[:, -band_width_bins:] = True
    interior = ~wall_band
    mean_near = float(rate_map[wall_band].mean())
    mean_far = float(rate_map[interior].mean())
    denom_robust = mean_near + mean_far
    b_robust = (mean_near - mean_far) / denom_robust if denom_robust > 0 else 0.0

    if not field_areas:
        return {
            'b': 0.0, 'c_M': 0.0, 'd_m': 0.0, 'n_fields': 0,
            'b_robust': b_robust, 'mean_near': mean_near, 'mean_far': mean_far,
            'field_areas': [], 'field_d_mins': [],
        }

    d_m = float(np.mean(field_d_mins))
    c_M = float(max(c_M_candidates))
    denom = c_M + d_m
    b = float((c_M - d_m) / denom) if denom > 0 else 0.0
    return {
        'b': b, 'c_M': c_M, 'd_m': d_m, 'n_fields': len(field_areas),
        'b_robust': b_robust, 'mean_near': mean_near, 'mean_far': mean_far,
        'field_areas': field_areas, 'field_d_mins': field_d_mins,
    }


def _empty_border_result():
    return {
        'b': 0.0, 'c_M': 0.0, 'd_m': 0.0, 'n_fields': 0,
        'b_robust': 0.0, 'mean_near': 0.0, 'mean_far': 0.0,
        'field_areas': [], 'field_d_mins': [],
    }


def compute_ebr(traj: dict, signals: np.ndarray,
                n_theta_bins: int = 36, n_dist_bins: int = 20) -> tuple:
    """Build egocentric boundary rate map (bearing × distance to nearest wall).

    Per the article's EBC procedure, but simplified: at each timestep we use
    the SINGLE nearest wall (one of N/S/E/W), with bearing and distance
    measured relative to the animal's head direction. Each spike contributes
    its rate to one (θ, d) bin. Result is a 2D rate map.

    Args:
        traj: dict with x, y, head_direction, d_N, d_S, d_E, d_W (2D).
        signals: (n_trials, n_steps) firing rates.
        n_theta_bins: number of bearing bins spanning (-π, π].
        n_dist_bins: number of distance bins spanning [0, ARENA_CM/√2].

    Returns:
        rate_map: (n_theta_bins, n_dist_bins) — mean firing rate per bin.
        bearing_edges, dist_edges: bin edges for plotting.
    """
    x, y, hd = traj['x'], traj['y'], traj['head_direction']
    d_N, d_S = traj['d_N'], traj['d_S']
    d_E, d_W = traj['d_E'], traj['d_W']
    A = config.ARENA_CM
    cx, cy = A / 2.0, A / 2.0

    # For each timestep, find the nearest wall and its bearing+distance
    d_min = np.minimum(np.minimum(d_N, d_S),
                       np.minimum(d_E, d_W))   # (n_trials, n_steps)
    is_N = (d_min == d_N)
    is_S = (d_min == d_S) & ~is_N
    is_E = (d_min == d_E) & ~is_N & ~is_S
    is_W = (d_min == d_W) & ~is_N & ~is_S & ~is_E

    # Allocentric bearing to each wall (from animal to wall midpoint on the wall)
    # For N wall: midpoint is (cx, A); bearing = atan2(A - y, cx - x)
    # For S wall: midpoint is (cx, 0); bearing = atan2(0 - y, cx - x)
    # For E wall: midpoint is (A, cy); bearing = atan2(cy - y, A - x)
    # For W wall: midpoint is (0, cy); bearing = atan2(cy - y, 0 - x)
    bearing_N = np.arctan2(A - y, cx - x)
    bearing_S = np.arctan2(0.0 - y, cx - x)
    bearing_E = np.arctan2(cy - y, A - x)
    bearing_W = np.arctan2(cy - y, 0.0 - x)

    # Egocentric bearing = allocentric - head_direction, wrapped
    def wrap(d): return np.arctan2(np.sin(d), np.cos(d))

    egb_N = wrap(bearing_N - hd)
    egb_S = wrap(bearing_S - hd)
    egb_E = wrap(bearing_E - hd)
    egb_W = wrap(bearing_W - hd)

    egbearing = np.where(is_N, egb_N,
                np.where(is_S, egb_S,
                 np.where(is_E, egb_E, egb_W)))
    egdist = d_min  # distance to the chosen wall

    bearing_edges = np.linspace(-np.pi, np.pi, n_theta_bins + 1)
    dist_max = A / np.sqrt(2)
    dist_edges = np.linspace(0.0, dist_max, n_dist_bins + 1)

    # Build 2D rate map
    rate_map = np.zeros((n_theta_bins, n_dist_bins), dtype=np.float64)
    occ = np.zeros((n_theta_bins, n_dist_bins), dtype=np.float64)

    egb_flat = egbearing.flatten()
    egd_flat = egdist.flatten()
    sig_flat = signals.flatten()

    # Vectorized bin assignment
    theta_idx = np.clip(np.searchsorted(bearing_edges, egb_flat) - 1,
                        0, n_theta_bins - 1)
    dist_idx = np.clip(np.searchsorted(dist_edges, egd_flat) - 1,
                       0, n_dist_bins - 1)
    np.add.at(rate_map, (theta_idx, dist_idx), sig_flat)
    np.add.at(occ, (theta_idx, dist_idx), 1.0)
    rate_map = np.where(occ > 0, rate_map / np.maximum(occ, 1.0), 0.0)
    return rate_map, bearing_edges, dist_edges


def ebc_mean_vector(rate_map: np.ndarray, bearing_edges: np.ndarray) -> complex:
    """Article's EBC Mean Vector (Methods, "Identification of egocentric boundary cells"):

        MV = sum_{θ,d} F_{θ,d} · exp(i·θ) / sum_{θ,d} F_{θ,d}

    Args:
        rate_map: (n_theta_bins, n_dist_bins) egocentric boundary rate map.
        bearing_edges: (n_theta_bins + 1,) bin edges for θ.

    Returns:
        MV: complex number; MVL = |MV|, preferred_bearing = angle(MV).
    """
    theta_centers = 0.5 * (bearing_edges[:-1] + bearing_edges[1:])
    Theta, _ = np.meshgrid(theta_centers, np.arange(rate_map.shape[1]), indexing='ij')
    F = rate_map
    z_sum = (F * np.exp(1j * Theta)).sum()
    total = F.sum()
    return z_sum / total if total > 0 else 0.0 + 0.0j


# ============================================================
# Visualization
# ============================================================

def visualize_dataset(dataset_path=None, trajectory_path=None,
                      output_dir=None, trial_idx=0):
    """Create 4×4 multi-panel figure: trajectory, inputs, target diagnostics."""

    ds_path = dataset_path or os.path.join(os.path.dirname(config.TRAJECTORY_HDF5),
                                           'dataset.h5')
    traj_path = trajectory_path or config.TRAJECTORY_HDF5
    out_dir = output_dir or config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    # ----- Load -----
    print(f"Loading trajectory from {traj_path}...")
    traj = load_trajectory_from_hdf5(traj_path)

    print(f"Loading dataset from {ds_path}...")
    ds = load_dataset_hdf5(ds_path)
    X = ds['X']            # (n_trials, n_steps, 21)
    Y = ds['Y']            # (n_trials, n_steps, 4)
    metadata = ds['metadata']

    if X.ndim != 3:
        raise ValueError(f"Expected X shape (n_trials, n_steps, 21), got {X.shape}")
    n_trials, n_steps, n_inputs = X.shape
    if n_inputs != config.N_INPUTS:
        raise ValueError(f"Expected {config.N_INPUTS} input channels, got {n_inputs}")
    if trial_idx < 0 or trial_idx >= n_trials:
        trial_idx = 0

    dt_ms = float(metadata.get('dt', config.DT))
    t_s = np.arange(n_steps, dtype=np.float32) * dt_ms / 1000.0

    # ----- Ground-truth derived variables (Long et al. formulas) -----
    CB_true = compute_cb(traj)        # (n_trials, n_steps)
    CD_true = compute_cd(traj)        # (n_trials, n_steps)

    # ----- Channel slices -----
    d_far = X[..., 0]
    d_near = X[..., 1]
    speed = X[..., 2]
    cb_cells = X[..., 3:3 + config.N_CB]               # (n_trials, n_steps, 8)
    cdhd_cells = X[..., 3 + config.N_CB:3 + config.N_CB + config.N_HD]  # (n_trials, n_steps, 8)
    cd_far = X[..., 3 + config.N_CB + config.N_HD]
    cd_near = X[..., -1]

    # ----- Apply article formulas: per-trial stats, then mean across trials -----
    # CB cells: MVL and preferred bearing (per cell, mean across trials)
    cb_mvl, cb_pref_rad = circular_mvl(cb_cells, CB_true)         # (8,)
    # CD×HD cells: MVL and preferred heading (over allocentric HD)
    cdhd_mvl, cdhd_pref_rad = circular_mvl(cdhd_cells, traj['head_direction'])
    # CD×HD cells: R² vs CD (positive-slope linear fit, like the article's CD cells)
    cdhd_R2, cdhd_slope = linear_fit_R2(cdhd_cells, CD_true)
    # cd_far / cd_near: R² vs CD
    cd_far_R2, cd_far_slope = linear_fit_R2(cd_far[..., None], CD_true)
    cd_near_R2, cd_near_slope = linear_fit_R2(cd_near[..., None], CD_true)

    # Border targets: per-trial spatial rate map → border score b
    border_targets = Y[..., 0], Y[..., 1], Y[..., 2], Y[..., 3]
    border_names = ['N', 'S', 'E', 'W']
    border_results = []
    ebc_mvls = []
    ebc_prefs = []
    ebr_maps = []  # EBR for each border cell
    rate_maps = []
    for k, (name, sig) in enumerate(zip(border_names, border_targets)):
        # Build per-trial rate maps, average across trials
        per_trial_maps = [compute_spatial_rate_map(
            {'x': traj['x'][t:t+1], 'y': traj['y'][t:t+1]}, sig[t]) for t in range(n_trials)]
        avg_map = np.mean(per_trial_maps, axis=0)
        bs = compute_border_score(avg_map)
        bs['name'] = name
        border_results.append(bs)
        rate_maps.append(avg_map)
        # EBR for the averaged signals
        sig_avg = sig.mean(axis=0)
        ebr, thedges, dedges = compute_ebr(traj, sig)
        ebr_maps.append(ebr)
        mv = ebc_mean_vector(ebr, thedges)
        ebc_mvls.append(abs(mv))
        ebc_prefs.append(np.angle(mv))

    # ----- Figure -----
    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

    # Row 0: trajectory, scalars, CB heatmap, CD×HD heatmap
    ax_traj = fig.add_subplot(gs[0, 0])
    for i in range(n_trials):
        ax_traj.plot(traj['x'][i], traj['y'][i], linewidth=0.3, alpha=0.5, color='gray')
    ax_traj.plot(traj['x'][trial_idx], traj['y'][trial_idx], linewidth=0.5, alpha=0.9, color='C0')
    ax_traj.scatter(traj['x'][trial_idx, 0], traj['y'][trial_idx, 0],
                    s=40, c='green', marker='o', label='start', zorder=5)
    ax_traj.set_xlim(0, config.ARENA_CM)
    ax_traj.set_ylim(0, config.ARENA_CM)
    ax_traj.set_aspect('equal')
    ax_traj.set_title(f'Trajectory ({n_trials} trials, trial {trial_idx} in blue)')
    ax_traj.set_xlabel('x (cm)')
    ax_traj.set_ylabel('y (cm)')

    ax_scal = fig.add_subplot(gs[0, 1])
    ax_scal.plot(t_s, d_far[trial_idx], label='d_far', linewidth=0.8)
    ax_scal.plot(t_s, d_near[trial_idx], label='d_near', linewidth=0.8)
    ax_scal.plot(t_s, speed[trial_idx], label='speed', linewidth=0.8)
    ax_scal.set_title('Scalar inputs (trial ' + str(trial_idx) + ')')
    ax_scal.set_xlabel('Time (s)')
    ax_scal.set_ylabel('Rate (Hz)')
    ax_scal.legend(fontsize=7)
    ax_scal.grid(True, alpha=0.3)

    ax_cb_hm = fig.add_subplot(gs[0, 2])
    step = max(1, n_steps // 2000)
    cb_show = cb_cells[trial_idx, ::step].T
    im = ax_cb_hm.imshow(cb_show, aspect='auto', origin='lower',
                         extent=[t_s[0], t_s[-1], 0, config.N_CB],
                         cmap='viridis', vmin=0, vmax=config.F_MAX_CB)
    ax_cb_hm.set_title(f'CB cells (8 channels, trial {trial_idx})')
    ax_cb_hm.set_xlabel('Time (s)')
    ax_cb_hm.set_ylabel('CB channel')
    plt.colorbar(im, ax=ax_cb_hm, label='Hz')

    ax_cdhd_hm = fig.add_subplot(gs[0, 3])
    cdhd_show = cdhd_cells[trial_idx, ::step].T
    im2 = ax_cdhd_hm.imshow(cdhd_show, aspect='auto', origin='lower',
                            extent=[t_s[0], t_s[-1], 0, config.N_HD],
                            cmap='viridis', vmin=0)
    ax_cdhd_hm.set_title(f'CD×HD cells (8 channels, trial {trial_idx})')
    ax_cdhd_hm.set_xlabel('Time (s)')
    ax_cdhd_hm.set_ylabel('CD×HD channel')
    plt.colorbar(im2, ax=ax_cdhd_hm, label='Hz')

    # Row 1: CB polar, CD×HD polar, cd_far vs CD, CB MVLs
    ax_cb_pol = fig.add_subplot(gs[1, 0], projection='polar')
    # Build mean rate per (CB-true) bin, per cell
    n_theta = 36
    bin_edges = np.linspace(-np.pi, np.pi, n_theta + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    CB_flat = CB_true[trial_idx]
    for k in range(config.N_CB):
        sums, _ = np.histogram(CB_flat, bins=bin_edges, weights=cb_cells[trial_idx, :, k])
        counts, _ = np.histogram(CB_flat, bins=bin_edges)
        mean_rate = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        ax_cb_pol.plot(bin_centers, mean_rate, linewidth=1.5,
                       label=f'k={k} pref={np.rad2deg(cb_pref_rad[k]):+.0f}°')
    ax_cb_pol.set_title('CB tuning curves (8 cells)\ncolor = cell, '
                        f'⟨MVL⟩ = {cb_mvl.mean():.2f}')
    ax_cb_pol.set_theta_zero_location('E')
    ax_cb_pol.set_theta_direction(1)
    ax_cb_pol.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.4, 1.1))

    ax_cdhd_pol = fig.add_subplot(gs[1, 1], projection='polar')
    hd_flat = traj['head_direction'][trial_idx]
    for k in range(config.N_HD):
        sums, _ = np.histogram(hd_flat, bins=bin_edges, weights=cdhd_cells[trial_idx, :, k])
        counts, _ = np.histogram(hd_flat, bins=bin_edges)
        mean_rate = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        ax_cdhd_pol.plot(bin_centers, mean_rate, linewidth=1.5,
                         label=f'k={k} pref={np.rad2deg(cdhd_pref_rad[k]):+.0f}°')
    ax_cdhd_pol.set_title('CD×HD tuning curves (8 cells)\n'
                          f'⟨MVL⟩ = {cdhd_mvl.mean():.2f}, ⟨R² vs CD⟩ = {cdhd_R2.mean():.2f}')
    ax_cdhd_pol.set_theta_zero_location('E')
    ax_cdhd_pol.set_theta_direction(1)
    ax_cdhd_pol.legend(fontsize=6, loc='upper right', bbox_to_anchor=(1.4, 1.1))

    ax_cdfar = fig.add_subplot(gs[1, 2])
    cd_flat = CD_true[trial_idx]
    n_cd_bins = 30
    cd_bin_edges = np.linspace(0, config.CD_MAX, n_cd_bins + 1)
    cd_bin_centers = 0.5 * (cd_bin_edges[:-1] + cd_bin_edges[1:])
    sums, _ = np.histogram(cd_flat, bins=cd_bin_edges, weights=cd_far[trial_idx])
    counts, _ = np.histogram(cd_flat, bins=cd_bin_edges)
    mean_far = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    sums, _ = np.histogram(cd_flat, bins=cd_bin_edges, weights=cd_near[trial_idx])
    mean_near = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    ax_cdfar.plot(cd_bin_centers, mean_far, 'o-', color='C3', label='cd_far')
    ax_cdfar.plot(cd_bin_centers, mean_near, 's-', color='C0', label='cd_near')
    ax_cdfar.set_xlabel('CD (cm)')
    ax_cdfar.set_ylabel('Rate (Hz)')
    ax_cdfar.set_title(f'CD cells vs CD\n'
                       f'R²: far={float(cd_far_R2[0]):.3f}, near={float(cd_near_R2[0]):.3f}')
    ax_cdfar.grid(True, alpha=0.3)
    ax_cdfar.legend(fontsize=8)

    ax_cb_mvl = fig.add_subplot(gs[1, 3])
    x_pos = np.arange(config.N_CB)
    ax_cb_mvl.bar(x_pos, cb_mvl, color='C2', alpha=0.7, label='MVL (CB)')
    ax_cb_mvl.set_xticks(x_pos)
    ax_cb_mvl.set_xticklabels([f'{np.rad2deg(cb_pref_rad[k]):+.0f}°'
                               for k in range(config.N_CB)], fontsize=8)
    ax_cb_mvl.set_ylim(0, 1)
    ax_cb_mvl.set_ylabel('MVL')
    ax_cb_mvl.set_title('CB cells: MVL & preferred bearing')
    ax_cb_mvl.axhline(cb_mvl.mean(), color='k', linestyle='--', alpha=0.5,
                      label=f'mean={cb_mvl.mean():.3f}')
    ax_cb_mvl.legend(fontsize=7)
    ax_cb_mvl.grid(True, alpha=0.3, axis='y')

    # Row 2: 4 border target spatial rate maps
    for k, (name, rmap) in enumerate(zip(border_names, rate_maps)):
        ax = fig.add_subplot(gs[2, k])
        im = ax.imshow(rmap.T, origin='lower', extent=[0, config.ARENA_CM, 0, config.ARENA_CM],
                       cmap='hot', vmin=0, vmax=config.F_MAX_BORDER)
        ax.set_aspect('equal')
        ax.set_title(f'Border_{name} target\n'
                     f'b_robust={border_results[k]["b_robust"]:+.2f}  '
                     f'b_art={border_results[k]["b"]:+.2f}  '
                     f'fields={border_results[k]["n_fields"]}')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        plt.colorbar(im, ax=ax, label='Hz')

    # Row 3: border score bar, EBC MVL bar, EBR polar (Border_N), diagnostics
    ax_bs = fig.add_subplot(gs[3, 0])
    b_robust_vals = [r['b_robust'] for r in border_results]
    b_art_vals = [r['b'] for r in border_results]
    x_pos = np.arange(len(border_names))
    w = 0.4
    ax_bs.bar(x_pos - w/2, b_robust_vals, width=w, color='C0', alpha=0.8,
              label='b_robust (1D)')
    ax_bs.bar(x_pos + w/2, b_art_vals, width=w, color='C3', alpha=0.8,
              label='b_art (Lever 2009)')
    ax_bs.set_xticks(x_pos)
    ax_bs.set_xticklabels(border_names)
    ax_bs.set_ylim(-1, 1)
    ax_bs.axhline(0, color='k', linewidth=0.5)
    ax_bs.set_ylabel('Border score b')
    ax_bs.set_title('Border score per target')
    ax_bs.legend(fontsize=7)
    ax_bs.grid(True, alpha=0.3, axis='y')

    ax_ebc = fig.add_subplot(gs[3, 1])
    ebc_mvl_vals = ebc_mvls
    ax_ebc.bar(border_names, ebc_mvl_vals, color=['C0', 'C1', 'C2', 'C3'], alpha=0.8)
    ax_ebc.set_ylim(0, 1)
    ax_ebc.set_ylabel('EBC MVL')
    ax_ebc.set_title('Egocentric boundary MV per border target\n'
                     'MV = Σ F·exp(iθ) / Σ F  (bearing to nearest wall)')
    ax_ebc.grid(True, alpha=0.3, axis='y')

    ax_ebr = fig.add_subplot(gs[3, 2], projection='polar')
    ebr_n = ebr_maps[0]
    thedges = np.linspace(-np.pi, np.pi, ebr_n.shape[0] + 1)
    dedges = np.linspace(0, config.ARENA_CM / np.sqrt(2), ebr_n.shape[1] + 1)
    the_centers = 0.5 * (thedges[:-1] + thedges[1:])
    r_centers = 0.5 * (dedges[:-1] + dedges[1:])
    Theta, R = np.meshgrid(the_centers, r_centers, indexing='ij')
    im_polar = ax_ebr.pcolormesh(Theta, R, ebr_n, cmap='hot', shading='auto')
    ax_ebr.set_theta_zero_location('E')
    ax_ebr.set_theta_direction(1)
    ax_ebr.set_title(f'EBR polar map: Border_N target\n'
                     f'EBC MVL={ebc_mvl_vals[0]:.2f}, '
                     f'pref_bearing={np.rad2deg(ebc_prefs[0]):+.0f}°',
                     fontsize=9)
    plt.colorbar(im_polar, ax=ax_ebr, label='Hz', pad=0.1)

    ax_diag = fig.add_subplot(gs[3, 3])
    ax_diag.axis('off')
    info = (
        f"Dataset: {ds_path}\n"
        f"Trials × Steps: {n_trials} × {n_steps}\n"
        f"DT: {dt_ms} ms, Arena: {config.ARENA_CM} cm\n"
        f"Inputs: {config.N_INPUTS} channels\n"
        f"Targets: 4 walls\n"
        f"Shapes: X={X.shape}, Y={Y.shape}\n"
        f"\n--- CB cells (n={config.N_CB}) ---\n"
        f"  ⟨MVL⟩ = {cb_mvl.mean():.3f}\n"
        f"  pref bearings (deg): "
        f"{', '.join(f'{np.rad2deg(cb_pref_rad[k]):+.0f}' for k in range(config.N_CB))}\n"
        f"  (target: 0, 45, 90, 135, 180, 225, 270, 315)\n"
        f"\n--- CD×HD cells (n={config.N_HD}) ---\n"
        f"  ⟨MVL over HD⟩ = {cdhd_mvl.mean():.3f}\n"
        f"  ⟨R² vs CD⟩ = {cdhd_R2.mean():.3f}\n"
        f"  ⟨slope vs CD⟩ = {cdhd_slope.mean():.4f}\n"
        f"\n--- CD cells ---\n"
        f"  cd_far:  R²={float(cd_far_R2[0]):.3f}, slope={float(cd_far_slope[0]):+.4f}\n"
        f"  cd_near: R²={float(cd_near_R2[0]):.3f}, slope={float(cd_near_slope[0]):+.4f}\n"
        f"\n--- Border targets ---\n"
        + "\n".join(
            f"  Border_{r['name']}: b_art={r['b']:+.3f}  "
            f"b_robust={r['b_robust']:+.3f}  "
            f"cM={r['c_M']:.1f}  dm={r['d_m']:.1f}  "
            f"fields={r['n_fields']}  EBC_MVL={ebc_mvls[k]:.3f}"
            for k, r in enumerate(border_results)
        )
    )
    ax_diag.text(0.02, 0.98, info, transform=ax_diag.transAxes,
                 fontsize=8, verticalalignment='top', fontfamily='monospace')

    plt.suptitle('Dataset diagnostics (Long et al. 2025 formulas)', fontsize=14, y=0.995)
    out_path = os.path.join(out_dir, 'dataset_preview.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print()
    print("--- Summary ---")
    print(f"CB MVL:       {cb_mvl}")
    print(f"CD×HD MVL:    {cdhd_mvl}")
    print(f"CD×HD R²:     {cdhd_R2}")
    print(f"cd_far R²:    {float(cd_far_R2[0]):.3f}, slope={float(cd_far_slope[0]):.4f}")
    print(f"cd_near R²:   {float(cd_near_R2[0]):.3f}, slope={float(cd_near_slope[0]):.4f}")
    print(f"Border scores (b_robust, b_art): " + ", ".join(
        f"{r['name']}=({r['b_robust']:+.3f}, {r['b']:+.3f})" for r in border_results))
    print(f"EBC MVLs:     " + ", ".join(f"{n}={v:.3f}" for n, v in zip(border_names, ebc_mvl_vals)))


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--trajectory", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--trial", type=int, default=0,
                        help="Trial index to highlight in time-series panels (default: 0)")
    args = parser.parse_args()
    visualize_dataset(args.dataset, args.trajectory, args.output_dir, args.trial)


if __name__ == '__main__':
    main()
