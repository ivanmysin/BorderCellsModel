"""Build parameter matrices for vectorized synapses."""

import numpy as np
import config
from utils.csv_loader import get_synapse_params_for_connection, get_neuron_ei, get_neuron_vr


# Unit indices: 0=Border_N, 1=Border_S, 2=Border_E, 3=Border_W, 4=Basket, 5=Axo
_UNIT_TYPES = [config.UNIT_TYPE[name] for name in config.UNIT_NAMES]

# Input channel structure (matches precompute_inputs in utils/inputs.py)
# 0:        d_far
# 1:        d_near
# 2:        speed
# 3..10:    CB_0..CB_7   (egocentric bearing to geometric center; Long et al. 2025)
# 11..18:   CDxHD_0..7   (allocentric head direction × positive CD slope)
# 19:       cd_far
# 20:       cd_near
_INPUT_NAMES = (
    ['d_far', 'd_near', 'speed']
    + [f'CB_{i}' for i in range(config.N_CB)]
    + [f'CDxHD_{i}' for i in range(config.N_HD)]
    + ['cd_far', 'cd_near']
)
assert len(_INPUT_NAMES) == config.N_INPUTS, (
    f"_INPUT_NAMES has {len(_INPUT_NAMES)} channels but "
    f"config.N_INPUTS={config.N_INPUTS}. Update _INPUT_NAMES in utils/params.py "
    f"to match precompute_inputs() in utils/inputs.py."
)


def _get_conn_key(src_type: str, tgt_type: str) -> str:
    return f'{src_type}→{tgt_type}'


def _get_syn_params(conn_key: str) -> dict:
    """Get raw synapse parameters from CSV for a connection key."""
    try:
        return get_synapse_params_for_connection(conn_key)
    except (ValueError, KeyError):
        if 'Basket' in conn_key or 'Axoaxonic' in conn_key:
            if 'Pyramidal' in conn_key or 'Input' in conn_key:
                return config.TM_SYN_DEFAULTS['Inh→Exc']
            return config.TM_SYN_DEFAULTS['Inh→Inh']
        return config.TM_SYN_DEFAULTS['Exc→Exc']


def _get_e_r(src_type: str, tgt_type: str) -> float:
    """Compute e_r for a connection."""
    pre_csv = config.NEURON_TYPE_MAP.get(src_type, src_type)
    post_csv = config.NEURON_TYPE_MAP.get(tgt_type, tgt_type)
    pre_ei = get_neuron_ei(pre_csv)
    Vr_post = get_neuron_vr(post_csv)
    E_r = 0.0 if pre_ei == 'e' else -75.0
    return 1.0 + E_r / abs(Vr_post)


def _get_e_r_input(tgt_type: str) -> float:
    """Compute e_r for input→target connections (all inputs are excitatory)."""
    # post_csv = config.NEURON_TYPE_MAP[tgt_type]
    # Vr_post = get_neuron_vr(post_csv)
    return 1.0 # + 0.0 / abs(Vr_post)  # E_r = 0 for exc inputs


def build_rec_gsyn_matrix() -> np.ndarray:
    """Build [6,6] gsyn_max matrix for recurrent connections.

    Attractor-style role assignment (deterministic):
      - Border self-excitation: B_X → B_X  (SELF_EXC)
      - Borders drive Basket:  B_X → Basket (B_TO_BASKET)
      - Basket inhibits all borders: Basket → B_X (BASKET_TO_B)
      - Basket self-inhibition: Basket → Basket (BASKET_SELF)
      - Axo globally inhibits borders: Axo → B_X (AXO_TO_B)
      - Axo self-inhibition: Axo → Axo (AXO_SELF)
      - All cross-border connections are 0 (WTA purely via Basket)

    Magnitudes are final gsyn_max values (NOT multiplied by
    GSYN_SCALE_DIMENSIONAL). Calibrated so I_syn ≈ 2-3 at 15 Hz
    steady state (A ≈ 0.018).
    """
    g = config.ATTRACTOR_GSYN
    m = np.zeros((config.N_POP_UNITS, config.N_POP_UNITS), dtype=np.float64)

    for i in range(4):
        m[i, i] = g['SELF_EXC']
        m[i, 4] = g['B_TO_BASKET']

    for j in range(4):
        m[4, j] = g['BASKET_TO_B']
    m[4, 4] = g['BASKET_SELF']

    for j in range(4):
        m[5, j] = g['AXO_TO_B']
    m[5, 5] = g['AXO_SELF']

    return m


def build_rec_tau_f_matrix() -> np.ndarray:
    """Build [6,6] tau_f matrix for recurrent connections."""
    m = np.zeros((config.N_POP_UNITS, config.N_POP_UNITS), dtype=np.float64)
    for i, src_name in enumerate(config.UNIT_NAMES):
        for j, tgt_name in enumerate(config.UNIT_NAMES):
            conn_key = _get_conn_key(_UNIT_TYPES[i], _UNIT_TYPES[j])
            p = _get_syn_params(conn_key)
            m[i, j] = p['tau_f']
    return m


def build_rec_tau_d_matrix() -> np.ndarray:
    """Build [6,6] tau_d matrix for recurrent connections."""
    m = np.zeros((config.N_POP_UNITS, config.N_POP_UNITS), dtype=np.float64)
    for i, src_name in enumerate(config.UNIT_NAMES):
        for j, tgt_name in enumerate(config.UNIT_NAMES):
            conn_key = _get_conn_key(_UNIT_TYPES[i], _UNIT_TYPES[j])
            p = _get_syn_params(conn_key)
            m[i, j] = p['tau_d']
    return m


def build_rec_tau_r_matrix() -> np.ndarray:
    """Build [6,6] tau_r matrix for recurrent connections."""
    m = np.zeros((config.N_POP_UNITS, config.N_POP_UNITS), dtype=np.float64)
    for i, src_name in enumerate(config.UNIT_NAMES):
        for j, tgt_name in enumerate(config.UNIT_NAMES):
            conn_key = _get_conn_key(_UNIT_TYPES[i], _UNIT_TYPES[j])
            p = _get_syn_params(conn_key)
            m[i, j] = p['tau_r']
    return m


def build_rec_Uinc_matrix() -> np.ndarray:
    """Build [6,6] Uinc matrix for recurrent connections."""
    m = np.zeros((config.N_POP_UNITS, config.N_POP_UNITS), dtype=np.float64)
    for i, src_name in enumerate(config.UNIT_NAMES):
        for j, tgt_name in enumerate(config.UNIT_NAMES):
            conn_key = _get_conn_key(_UNIT_TYPES[i], _UNIT_TYPES[j])
            p = _get_syn_params(conn_key)
            m[i, j] = p['Uinc']
    return m


def build_rec_pconn_matrix() -> np.ndarray:
    """Build [6,6] pconn matrix for recurrent connections (all-to-all)."""
    return np.ones((config.N_POP_UNITS, config.N_POP_UNITS), dtype=np.float64)


def build_rec_e_r_matrix() -> np.ndarray:
    """Build [6,6] e_r matrix for recurrent connections."""
    m = np.zeros((config.N_POP_UNITS, config.N_POP_UNITS), dtype=np.float64)
    for i, src_name in enumerate(config.UNIT_NAMES):
        for j, tgt_name in enumerate(config.UNIT_NAMES):
            m[i, j] = _get_e_r(_UNIT_TYPES[i], _UNIT_TYPES[j])
    return m


def _get_input_conn_key(inp_name: str) -> str:
    """Get the synapse connection key for an input channel."""
    return f'Input→Pyramidal'  # all inputs treated as excitatory


def _input_targets(inp_name: str) -> list:
    """Return list of target unit indices for an input channel."""
    if inp_name == 'd_far':
        return list(range(4))  # border only
    elif inp_name == 'd_near':
        return list(range(4))
    elif inp_name == 'speed':
        return list(range(4))
    else:  # HD channels
        return list(range(4))


def build_inp_gsyn_matrix() -> np.ndarray:
    """Build [21,6] gsyn_max matrix for input→population connections.

    Per-channel inductive bias (matches utils/inputs.py::precompute_inputs):

    - d_far (idx 0):  primary drive for Axo (off-wall global inhibitor) and
                      weak drive for Basket — see ATTRACTOR_GSYN['DFAR_TO_*'].
                      All borders also get a weak base drive.
    - d_near (idx 1), speed (idx 2): weak base drive to all borders.
    - CB_i (idx 3..10): egocentric bearing to geometric center. For a Border_j
                      cell, the most informative CB channel is the one whose
                      θ_pref matches the allocentric bearing to the center
                      WHEN the animal is at wall j — i.e. WALL_ANGLES[j] + π.
                      Uses Gaussian similarity with σ=HD_SIGMA_RAD.
    - CDxHD_i (idx 11..18): allocentric HD × positive CD slope. Behaves like
                      the previous HD population, so use the original rule:
                      similarity to WALL_ANGLES[j] (allocentric direction TO
                      wall j from the animal at the wall).
    - cd_far (idx 19): positive drive to all borders (high when far from
                      center = at a wall).
    - cd_near (idx 20): weak drive to all borders (high at center, low at
                      walls; the network will learn the right sign).
    """
    m = np.zeros((config.N_INPUTS, config.N_POP_UNITS), dtype=np.float64)
    g_atr = config.ATTRACTOR_GSYN
    conn_key = 'Input→Pyramidal'
    p = _get_syn_params(conn_key)
    g_base = p['gsyn_max'] * config.GSYN_SCALE_DIMENSIONAL
    sigma2 = 2 * config.HD_SIGMA_RAD ** 2

    for i, inp_name in enumerate(_INPUT_NAMES):
        if inp_name == 'd_far':
            for j in range(4):
                m[i, j] = max(0.001, g_base)
            m[i, 4] = g_atr['DFAR_TO_BASKET']
            m[i, 5] = g_atr['DFAR_TO_AXO']
        elif inp_name.startswith('CB_'):
            cb_idx = int(inp_name.split('_')[1])
            cb_angle = np.deg2rad(config.THETA_PREF_CB[cb_idx])
            for j in range(4):
                # allocentric bearing to center when animal is at wall j
                center_bearing = (config.WALL_ANGLES[j] + np.pi) % (2 * np.pi)
                diff = (cb_angle - center_bearing + np.pi) % (2 * np.pi) - np.pi
                g_dir = np.exp(-diff ** 2 / sigma2)
                m[i, j] = max(0.001, g_base * g_dir)
        elif inp_name.startswith('CDxHD_'):
            hd_idx = int(inp_name.split('_')[1])
            hd_angle = np.deg2rad(config.THETA_PREF_HD[hd_idx])
            for j in range(4):
                wall_angle = config.WALL_ANGLES[j]
                diff = (hd_angle - wall_angle + np.pi) % (2 * np.pi) - np.pi
                g_dir = np.exp(-diff ** 2 / sigma2)
                m[i, j] = max(0.001, g_base * g_dir)
        else:
            # d_near, speed, cd_far, cd_near → weak base drive to all borders
            for j in _input_targets(inp_name):
                m[i, j] = max(0.001, g_base)
    return m


def build_inp_tau_f_matrix() -> np.ndarray:
    """Build [21,6] tau_f matrix for input connections."""
    p = _get_syn_params('Input→Pyramidal')
    m = np.full((config.N_INPUTS, config.N_POP_UNITS), p['tau_f'], dtype=np.float64)
    return m


def build_inp_tau_d_matrix() -> np.ndarray:
    """Build [21,6] tau_d matrix for input connections."""
    p = _get_syn_params('Input→Pyramidal')
    m = np.full((config.N_INPUTS, config.N_POP_UNITS), p['tau_d'], dtype=np.float64)
    return m


def build_inp_tau_r_matrix() -> np.ndarray:
    """Build [21,6] tau_r matrix for input connections."""
    p = _get_syn_params('Input→Pyramidal')
    m = np.full((config.N_INPUTS, config.N_POP_UNITS), p['tau_r'], dtype=np.float64)
    return m


def build_inp_Uinc_matrix() -> np.ndarray:
    """Build [21,6] Uinc matrix for input connections."""
    m = np.zeros((config.N_INPUTS, config.N_POP_UNITS), dtype=np.float64) + 0.5
    p = _get_syn_params('Input→Pyramidal')
    for i, inp_name in enumerate(_INPUT_NAMES):
        for j in _input_targets(inp_name):
            m[i, j] = p['Uinc']
    return m


def build_inp_pconn_matrix() -> np.ndarray:
    """Build [21,6] pconn matrix for input connections (all-to-all)."""
    return np.ones((config.N_INPUTS, config.N_POP_UNITS), dtype=np.float64)


def build_inp_e_r_matrix() -> np.ndarray:
    """Build [21,6] e_r matrix for input connections."""
    m = np.zeros((config.N_INPUTS, config.N_POP_UNITS), dtype=np.float64)
    for j, tgt_name in enumerate(config.UNIT_NAMES):
        e_r = _get_e_r_input(_UNIT_TYPES[j])
        for i, inp_name in enumerate(_INPUT_NAMES):
            m[i, j] = e_r
    return m


def build_pop_params() -> dict:
    """Build per-unit population parameters for IzhikevichMeanField (6 units).

    Returns dict with keys: tau_pop, alpha, a, b, w_jump, Delta_I, I_ext
    Each value is a list of 6 floats.
    """
    from utils.csv_loader import get_izhikevich_dimensionless_params

    per_unit = {k: [] for k in ['tau_pop', 'alpha', 'a', 'b', 'w_jump', 'Delta_I', 'I_ext']}
    for name in config.UNIT_NAMES:
        ntype = config.UNIT_TYPE[name]
        p = get_izhikevich_dimensionless_params(config.NEURON_TYPE_MAP[ntype])
        for k in per_unit:
            per_unit[k].append(p[k])
    return per_unit
