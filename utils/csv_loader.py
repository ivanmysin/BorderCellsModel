"""Load parameters from CSV files and map to neuraltide formats."""

import pandas as pd
import config


def load_neuron_csv():
    """Load neuron parameters CSV."""
    return pd.read_csv(config.NEURON_PARAMS_CSV)


def load_synapse_csv():
    """Load synapse parameters CSV."""
    return pd.read_csv(config.SYNAPSE_PARAMS_CSV)


def _get_neuron_row(neuron_type: str) -> dict:
    """Extract parameters for a specific neuron type."""
    df = load_neuron_csv()
    row = df[df["Neuron Type"] == neuron_type]
    if row.empty:
        raise ValueError(f"Neuron type '{neuron_type}' not found in CSV")
    return row.iloc[0].to_dict()


def get_izhikevich_dimensional_params(neuron_type: str) -> dict:
    """Convert CSV Izhikevich params to neuraltide dimensional MPR params.

    Returns: {V_rest, V_T, V_peak, V_reset, Cm, K, A, B, W_jump, Delta_I, I_ext}
    """
    row = _get_neuron_row(neuron_type)
    is_exc = row.get("E/I", "e").strip() == "e"
    params = {
        "V_rest": float(row["Izh Vr"]),
        "V_T": float(row["Izh Vt"]),
        "V_peak": float(row["Izh Vpeak"]),
        "V_reset": float(row["Izh Vmin"]),
        "Cm": float(row["Izh C"]),
        "K": float(row["Izh k"]),
        "A": float(row["Izh a"]),
        "B": float(row["Izh b"]),
        "W_jump": float(row["Izh d"]),
        "Delta_I": config.DELTA_I_DEFAULT,
        "I_ext": config.I_EXT_DEFAULT_EXC if is_exc else config.I_EXT_DEFAULT_INH,
    }
    return params


def get_synapse_params(src_subregion: str, src_type: str,
                       tgt_subregion: str, tgt_type: str) -> dict:
    """Find synapse parameters matching source→target in CSV.

    Returns: {gsyn_max, tau_f, tau_d, tau_r, Uinc}
    """
    df = load_synapse_csv()
    match = df[
        (df["Source Subregion"] == src_subregion) &
        (df["Presynaptic Neuron Type"] == src_type) &
        (df["Target Subregion"] == tgt_subregion) &
        (df["Postsynaptic Neuron Type"] == tgt_type)
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "gsyn_max": float(row["g"]),
        "tau_d": float(row["tau_d"]),
        "tau_r": float(row["tau_r"]),
        "tau_f": float(row["tau_f"]),
        "Uinc": float(row["u"]),
    }


def get_synapse_params_for_connection(conn_key: str) -> dict:
    """Get synapse parameters for a named connection type.

    conn_key: one of "Pyramidal→Pyramidal", "Basket→Pyramidal", etc.
    Returns {gsyn_max, tau_f, tau_d, tau_r, Uinc}
    """
    key = config.SYNAPSE_TYPE_MAP.get(conn_key)
    if key is None:
        raise ValueError(f"Unknown connection key: {conn_key}")
    result = get_synapse_params(*key)
    if result is None:
        # fallback to defaults
        if "Basket" in conn_key or "Axoaxonic" in conn_key:
            if "Exc" in conn_key or "Pyramidal" in conn_key or "Input" in conn_key:
                result = config.TM_SYN_DEFAULTS["Inh→Exc"]
            else:
                result = config.TM_SYN_DEFAULTS["Inh→Inh"]
        else:
            result = config.TM_SYN_DEFAULTS["Exc→Exc"]
    return result
