"""Build parameter matrices for vectorized synapses."""

import numpy as np
import config
from utils.csv_loader import get_synapse_params_for_connection, get_neuron_ei, get_neuron_vr


# Unit indices: 0=Border_N, 1=Border_S, 2=Border_E, 3=Border_W, 4=Basket, 5=Axo
_UNIT_TYPES = [config.UNIT_TYPE[name] for name in config.UNIT_NAMES]

# Input channel structure
_INPUT_NAMES = (
    ['d_far', 'd_near', 'speed']
    + [f'HD_{i}' for i in range(config.N_HD)]
)
assert len(_INPUT_NAMES) == config.N_INPUTS


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
    post_csv = config.NEURON_TYPE_MAP[tgt_type]
    Vr_post = get_neuron_vr(post_csv)
    return 1.0 + 0.0 / abs(Vr_post)  # E_r = 0 for exc inputs


def build_rec_gsyn_matrix() -> np.ndarray:
    """Build [6,6] gsyn_max matrix for recurrent connections."""
    m = np.zeros((config.N_POP_UNITS, config.N_POP_UNITS), dtype=np.float64)
    for i, src_name in enumerate(config.UNIT_NAMES):
        for j, tgt_name in enumerate(config.UNIT_NAMES):
            conn_key = _get_conn_key(_UNIT_TYPES[i], _UNIT_TYPES[j])
            p = _get_syn_params(conn_key)
            g = p['gsyn_max'] * config.GSYN_SCALE_DIMENSIONAL
            g *= (1.0 + np.random.uniform(-0.3, 0.3))
            m[i, j] = max(0.001, g)
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

    For HD channels (indices 3..20), gsyn_max is direction-similarity weighted:
    Border_j gets a strong connection from HD cells whose preferred direction
    is close to WALL_ANGLES[j]. This gives the model the right inductive bias
    to differentiate border cells via the history of HD activity.
    """
    m = np.zeros((config.N_INPUTS, config.N_POP_UNITS), dtype=np.float64)
    for i, inp_name in enumerate(_INPUT_NAMES):
        conn_key = 'Input→Pyramidal'
        p = _get_syn_params(conn_key)
        g_base = p['gsyn_max'] * config.GSYN_SCALE_DIMENSIONAL

        if inp_name.startswith('HD_'):
            hd_idx = int(inp_name.split('_')[1])
            hd_angle = np.deg2rad(config.THETA_PREF[hd_idx])
            for j, unit_name in enumerate(config.UNIT_NAMES[:4]):
                wall_angle = config.WALL_ANGLES[j]
                diff = (hd_angle - wall_angle + np.pi) % (2 * np.pi) - np.pi
                g_dir = np.exp(-diff ** 2
                               / (2 * config.HD_SIGMA_RAD ** 2))
                g = g_base * g_dir * (1.0 + np.random.uniform(-0.3, 0.3))
                m[i, j] = max(0.001, g)
        else:
            g = g_base * (1.0 + np.random.uniform(-0.3, 0.3))
            for j in _input_targets(inp_name):
                m[i, j] = max(0.001, g)
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
    m = np.zeros((config.N_INPUTS, config.N_POP_UNITS), dtype=np.float64)
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
            if j in _input_targets(inp_name):
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
