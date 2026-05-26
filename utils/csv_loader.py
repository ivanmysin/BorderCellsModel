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


def get_izhikevich_dimensionless_params(neuron_type: str) -> dict:
    """Compute dimensionless MPR parameters from CSV Izhikevich 2003 values.

    Returns a dict suitable for IzhikevichMeanField in dimensionless mode:
        {tau_pop, alpha, a, b, w_jump, Delta_I, I_ext}

    tau_pop and alpha are derived from CSV structural params (Cm, K, V_rest, V_T)
    using the same formulas as neuraltide's internal conversion.
    a, b, w_jump, Delta_I, I_ext use config-stable targets with CSV-derived
    tau_pop scaling where appropriate.
    """
    row = _get_neuron_row(neuron_type)
    is_exc = row.get("E/I", "e").strip() == "e"

    K   = float(row["Izh k"])
    Vr  = float(row["Izh Vr"])
    VT  = float(row["Izh Vt"])
    Cm  = float(row["Izh C"])

    tau_pop = Cm / (K * abs(Vr))
    alpha   = 1.0 + VT / abs(Vr)

    alpha = max(alpha, config.MPR_ALPHA_MIN)
    tau_pop = max(tau_pop, config.MPR_TAU_POP_MIN)

    a_tgt = config.MPR_A_RS if is_exc else config.MPR_A_FS
    b_tgt = config.MPR_B_RS if is_exc else config.MPR_B_FS
    wj_tgt = config.MPR_WJ_RS if is_exc else config.MPR_WJ_FS
    ie_tgt = config.I_EXT_DIMENSIONLESS_RS if is_exc else config.I_EXT_DIMENSIONLESS_FS

    return {
        "tau_pop": tau_pop,
        "alpha":   alpha,
        "a":       a_tgt,
        "b":       b_tgt,
        "w_jump":  wj_tgt,
        "Delta_I": config.DELTA_I_DIMENSIONLESS,
        "I_ext":   ie_tgt,
    }


def get_izhikevich_dimensional_params(neuron_type: str) -> dict:
    """Convert CSV Izhikevich params to neuraltide dimensional MPR params.

    Returns: {V_rest, V_T, V_peak, V_reset, Cm, K, A, B, W_jump, Delta_I, I_ext}
    with values that yield the same dimensionless equivalents as
    get_izhikevich_dimensionless_params() after neuraltide's internal conversion.
    """
    row = _get_neuron_row(neuron_type)
    is_exc = row.get("E/I", "e").strip() == "e"

    Vr  = float(row["Izh Vr"])
    VT  = float(row["Izh Vt"])
    Vp  = float(row["Izh Vpeak"])
    Vmin = float(row["Izh Vmin"])
    K   = float(row["Izh k"])

    # Compute target dimensionless params
    dl = get_izhikevich_dimensionless_params(neuron_type)
    tau_tgt = dl["tau_pop"]
    alpha_tgt = dl["alpha"]

    # neuraltide conversion:
    #   tau_pop = Cm / (K * |Vr|)  →  Cm = tau_tgt * K * |Vr|
    #   alpha   = 1 + V_T / |Vr|  →  V_T = (alpha_tgt - 1) * |Vr|
    Cm = tau_tgt * K * abs(Vr)
    VT_eff = (alpha_tgt - 1.0) * abs(Vr)

    k_vr = K * abs(Vr)
    k_vr2 = k_vr * abs(Vr)

    A = dl["a"] * k_vr / Cm
    B = dl["b"] * k_vr
    W_jump = dl["w_jump"] * k_vr2
    I_ext = dl["I_ext"] * k_vr2
    Delta_I = dl["Delta_I"] * k_vr2

    return {
        "V_rest":  Vr,
        "V_T":     VT_eff,
        "V_peak":  Vp,
        "V_reset": Vmin,
        "Cm":      Cm,
        "K":       K,
        "A":       A,
        "B":       B,
        "W_jump":  W_jump,
        "Delta_I": Delta_I,
        "I_ext":   I_ext,
    }


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
        if "Basket" in conn_key or "Axoaxonic" in conn_key:
            if "Exc" in conn_key or "Pyramidal" in conn_key or "Input" in conn_key:
                result = config.TM_SYN_DEFAULTS["Inh→Exc"]
            else:
                result = config.TM_SYN_DEFAULTS["Inh→Inh"]
        else:
            result = config.TM_SYN_DEFAULTS["Exc→Exc"]
    return result
