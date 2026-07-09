"""Parameter matrices for sub-models (Phase 1: per-pyramidal 3-unit models).

Each sub-model has 3 units: one pyramidal [Border_X, Basket, Axo].
Input channels: 21 real (d_far, d_near, speed, HD×18) + 3 teacher inputs
(ideal targets of the OTHER three pyramids, computed as
f_max_border * exp(-d_other / lambda)).

Used by train_phase1.py. Lifter logic (Phase 2 mapping) lives in train_phase2.py.
"""

import numpy as np

import config
from utils.params import (
    build_pop_params,
    build_rec_gsyn_matrix, build_rec_tau_f_matrix,
    build_rec_tau_d_matrix, build_rec_tau_r_matrix,
    build_rec_Uinc_matrix, build_rec_pconn_matrix, build_rec_e_r_matrix,
    build_inp_gsyn_matrix, build_inp_tau_f_matrix,
    build_inp_tau_d_matrix, build_inp_tau_r_matrix,
    build_inp_Uinc_matrix, build_inp_pconn_matrix, build_inp_e_r_matrix,
)
from utils.csv_loader import (
    get_synapse_params_for_connection,
    get_neuron_vr,
)


N_SUB_INPUTS = config.N_INPUTS + 3  # 24: 21 real + 3 teacher channels


def _pyr_to_target_synapse(target_unit_name: str) -> dict:
    """Build TM params for Pyramidal→target (used by teacher inputs)."""
    try:
        p = get_synapse_params_for_connection(f"Pyramidal→{config.UNIT_TYPE[target_unit_name]}")
        g = p["gsyn_max"]
        tf_val = p["tau_f"]
        td = p["tau_d"]
        tr = p["tau_r"]
        u = p["Uinc"]
    except (ValueError, KeyError):
        d = config.TM_SYN_DEFAULTS["Exc→Exc"]
        g, td, tr, tf_val, u = (
            d["gsyn_max"], d["tau_d"], d["tau_r"], d["tau_f"], d["Uinc"],
        )

    g_scaled = g * config.GSYN_SCALE_DIMENSIONAL
    g_perturbed = g_scaled * (1.0 + np.random.uniform(-0.3, 0.3))

    Vr_post = get_neuron_vr(config.NEURON_TYPE_MAP[config.UNIT_TYPE[target_unit_name]])
    e_r = 1.0 + 0.0 / abs(Vr_post)  # Pyramidal is excitatory, E_r = 0

    return {
        "gsyn_max": max(0.001, g_perturbed),
        "tau_f": float(tf_val),
        "tau_d": float(td),
        "tau_r": float(tr),
        "Uinc": float(u),
        "pconn": 1.0,
        "e_r": e_r,
    }


def build_submodel_params(X_idx: int) -> dict:
    """Build params for a 3-unit sub-model focused on Border_X.

    X_idx ∈ {0, 1, 2, 3} — index of the border cell being modelled
    (0=Border_N, 1=Border_S, 2=Border_E, 3=Border_W).

    Sub-model unit order: [Border_X, Basket, Axo] (pyramidal at index 0).
    Sub-model input order: 21 real channels (same as full model) + 3 teacher
    channels (other pyramids' ideal targets, in increasing index order).

    Returns dict with the same keys as utils.params.gather_params():
        alpha, a, b, w_jump, tau_pop, I_ext, Delta_I: shape (3,)
        gsyn_max, tau_f, tau_d, tau_r, Uinc, pconn, e_r: shape (27, 3)
    """
    assert X_idx in (0, 1, 2, 3), f"X_idx must be 0..3, got {X_idx}"

    other_pyramids = [i for i in range(4) if i != X_idx]
    full_idx = [X_idx, config.UNIT_IDX["Basket"], config.UNIT_IDX["Axo"]]

    pop = build_pop_params()
    alpha   = np.asarray([pop["alpha"][i]   for i in full_idx], dtype=np.float32)
    a       = np.asarray([pop["a"][i]       for i in full_idx], dtype=np.float32)
    b       = np.asarray([pop["b"][i]       for i in full_idx], dtype=np.float32)
    w_jump  = np.asarray([pop["w_jump"][i]  for i in full_idx], dtype=np.float32)
    tau_pop = np.asarray([pop["tau_pop"][i] for i in full_idx], dtype=np.float32)
    Delta_I = np.asarray([pop["Delta_I"][i] for i in full_idx], dtype=np.float32)
    I_ext   = np.asarray([pop["I_ext"][i]   for i in full_idx], dtype=np.float32).copy()
    I_ext[0] += np.float32(np.random.uniform(-0.1, 0.1))

    full_rec_gsyn = build_rec_gsyn_matrix()
    rec_slices = (
        full_rec_gsyn[np.ix_(full_idx, full_idx)],
        build_rec_tau_f_matrix()[np.ix_(full_idx, full_idx)],
        build_rec_tau_d_matrix()[np.ix_(full_idx, full_idx)],
        build_rec_tau_r_matrix()[np.ix_(full_idx, full_idx)],
        build_rec_Uinc_matrix()[np.ix_(full_idx, full_idx)],
        build_rec_pconn_matrix()[np.ix_(full_idx, full_idx)],
        build_rec_e_r_matrix()[np.ix_(full_idx, full_idx)],
    )

    full_inp_gsyn = build_inp_gsyn_matrix()
    inp_slices = (
        full_inp_gsyn[:, full_idx],
        build_inp_tau_f_matrix()[:, full_idx],
        build_inp_tau_d_matrix()[:, full_idx],
        build_inp_tau_r_matrix()[:, full_idx],
        build_inp_Uinc_matrix()[:, full_idx],
        build_inp_pconn_matrix()[:, full_idx],
        build_inp_e_r_matrix()[:, full_idx],
    )

    teacher_rows = []
    for t_idx, other_pyr in enumerate(other_pyramids):
        _ = other_pyr  # teacher order is fixed by other_pyramids index sequence
        for unit_full_idx in full_idx:
            target_name = config.UNIT_NAMES[unit_full_idx]
            teacher_rows.append(_pyr_to_target_synapse(target_name))

    teacher_count = len(other_pyramids)
    unit_count = len(full_idx)

    def _stack_teacher(attr: str) -> np.ndarray:
        return np.asarray(
            [teacher_rows[t * unit_count + u][attr]
             for t in range(teacher_count)
             for u in range(unit_count)],
            dtype=np.float64,
        ).reshape(teacher_count, unit_count)

    teacher_mats = (
        _stack_teacher("gsyn_max"),
        _stack_teacher("tau_f"),
        _stack_teacher("tau_d"),
        _stack_teacher("tau_r"),
        _stack_teacher("Uinc"),
        _stack_teacher("pconn"),
        _stack_teacher("e_r"),
    )

    gsyn_max = np.vstack([rec_slices[0], inp_slices[0], teacher_mats[0]]).astype(np.float32)
    tau_f    = np.vstack([rec_slices[1], inp_slices[1], teacher_mats[1]]).astype(np.float32)
    tau_d    = np.vstack([rec_slices[2], inp_slices[2], teacher_mats[2]]).astype(np.float32)
    tau_r    = np.vstack([rec_slices[3], inp_slices[3], teacher_mats[3]]).astype(np.float32)
    Uinc     = np.vstack([rec_slices[4], inp_slices[4], teacher_mats[4]]).astype(np.float32)
    pconn    = np.vstack([rec_slices[5], inp_slices[5], teacher_mats[5]]).astype(np.float32)
    e_r      = np.vstack([rec_slices[6], inp_slices[6], teacher_mats[6]]).astype(np.float32)

    return {
        "alpha": alpha, "a": a, "b": b, "w_jump": w_jump,
        "tau_pop": tau_pop, "I_ext": I_ext, "Delta_I": Delta_I,
        "gsyn_max": gsyn_max, "tau_f": tau_f, "tau_d": tau_d, "tau_r": tau_r,
        "Uinc": Uinc, "pconn": pconn, "e_r": e_r,
    }


def augment_with_teachers(X: np.ndarray, Y: np.ndarray, X_idx: int) -> np.ndarray:
    """Append the OTHER 3 pyramids' target columns to X as teacher inputs.

    Args:
        X: shape (n_batches, T, 21) — real inputs only.
        Y: shape (n_batches, T, 4) — targets for N, S, E, W in that order.
        X_idx: 0..3 — which border cell is the model's "student".

    Returns:
        X_aug: shape (n_batches, T, 24) — real inputs + 3 teacher inputs.
            Teacher channels are Y columns of the OTHER 3 pyramids, in
            increasing-index order (so X_idx=0 gets [S, E, W] etc.).
    """
    assert X.shape[-1] == config.N_INPUTS, f"X must have {config.N_INPUTS} channels"
    assert Y.shape[-1] == 4, "Y must have 4 channels (N, S, E, W)"
    other = [i for i in range(4) if i != X_idx]
    teachers = Y[..., other]
    return np.concatenate([X, teachers.astype(X.dtype)], axis=-1)


def extract_target_for(Y: np.ndarray, X_idx: int) -> np.ndarray:
    """Pull out the column for pyramidal X_idx from a targets array."""
    return Y[..., X_idx:X_idx + 1].astype(np.float32)
