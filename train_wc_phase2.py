"""Phase 2 (WC): lift 4 Wilson-Cowan sub-models into the full 6-unit model.

Reads weights from ``results/phase1_wc/{border_N,border_S,border_E,
border_W}_submodel_vars.npz`` (produced by ``train_wc_phase1.py``),
assembles them into the full-model variable tensors via ``build_model``
from the untouched ``train_wc_nonpsyns.py``, then saves the resulting
weights to ``results/phase1_wc/lifted.weights.h5``. The user can then
resume full training with:

    python train_wc_nonpsyns.py --resume results/phase1_wc/lifted.weights.h5

Lifting strategy (defaults, mirrors Izhikevich train_phase2.py):
- pyramidal self-recurrent Border_X -> Border_X: from sub-model X recurrent.
- Border_Y -> Border_X (incoming to X): from sub-model X teacher row for Y.
- Border_X -> Border_Y (outgoing from X): from sub-model Y teacher row for X.
- Border_X <-> Basket/Axo (both directions): from sub-model X recurrent.
- Basket <-> Axo self & mutual, real-input synapses to Basket/Axo:
  averaged across the 4 sub-models.
- I_ext: per-Border uses its own sub-model; Basket/Axo use the average.

Usage:
    python train_wc_phase2.py [--phase1-dir DIR] [--out PATH]
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

import config
try:
    from train_wc_nonpsyns import (
        build_model,
        WilsonCowanNetwork,
        _softplus,
        _inv_softplus_np,
    )
except ModuleNotFoundError as e:
    sys.stderr.write(
        f"ERROR: {e}\n"
        f"  train_wc_phase2.py expects train_wc_nonpsyns.py next to it.\n"
    )
    raise

tf.get_logger().setLevel("ERROR")

PYRAMIDAL_NAMES = ["border_N", "border_S", "border_E", "border_W"]


def _load_submodel_vars(phase1_dir):
    """Load 4 sub-model vars npz. Returns dict {X_idx: {var_name: array}}.

    Each sub-model npz contains both the theta values
    (I_ext, theta_gsyn, theta_tau_1, theta_tau_2) and the physical
    values (gsyn_max, tau_1, tau_2) so the lifter can use either.
    """
    out = {}
    for X_idx, name in enumerate(PYRAMIDAL_NAMES):
        path = os.path.join(phase1_dir, f"{name}_submodel_vars.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run train_wc_phase1.py first.")
        data = np.load(path)
        out[X_idx] = {k: data[k] for k in data.files}
    return out


def _lift_matrix_var(sub_vars, var_name):
    """Lift a (pre, post) shape synapse param from 4 sub-models to full model.

    Sub-model pre layout:    [Border_X, Basket, Axo (3); real_inputs(21);
                              teacher(3)]  -> 27 rows
    Sub-model post layout:   [Border_X, Basket, Axo]
    Full-model pre layout:   [Border_N, Border_S, Border_E, Border_W,
                              Basket, Axo (6); real_inputs(21)]  -> 27 rows
    Full-model post layout:  [Border_N, Border_S, Border_E, Border_W,
                              Basket, Axo]
    """
    full_pre = config.N_POP_UNITS + config.N_INPUTS
    full_post = config.N_POP_UNITS
    full = np.zeros((full_pre, full_post), dtype=np.float32)

    rec_end_local = 3
    inp_end_local = rec_end_local + config.N_INPUTS
    teacher_off_local = inp_end_local

    for src_full in range(config.N_POP_UNITS):
        for tgt_full in range(config.N_POP_UNITS):
            if src_full < 4 and tgt_full < 4:
                if src_full == tgt_full:
                    val = sub_vars[tgt_full][var_name][0, 0]
                else:
                    other_tgt = [i for i in range(4) if i != tgt_full]
                    teacher_row = other_tgt.index(src_full)
                    val = sub_vars[tgt_full][var_name][
                        teacher_off_local + teacher_row, 0]
            elif tgt_full < 4 and src_full in (4, 5):
                val = sub_vars[tgt_full][var_name][src_full - 3, 0]
            elif tgt_full == 4 and src_full < 4:
                val = sub_vars[src_full][var_name][0, 1]
            elif tgt_full == 4 and src_full == 4:
                val = float(np.mean(
                    [sub_vars[X][var_name][1, 1] for X in range(4)]))
            elif tgt_full == 4 and src_full == 5:
                val = float(np.mean(
                    [sub_vars[X][var_name][2, 1] for X in range(4)]))
            elif tgt_full == 5 and src_full < 4:
                val = sub_vars[src_full][var_name][0, 2]
            elif tgt_full == 5 and src_full == 4:
                val = float(np.mean(
                    [sub_vars[X][var_name][1, 2] for X in range(4)]))
            elif tgt_full == 5 and src_full == 5:
                val = float(np.mean(
                    [sub_vars[X][var_name][2, 2] for X in range(4)]))
            else:
                raise RuntimeError(
                    f"unhandled src={src_full}, tgt={tgt_full}")
            full[src_full, tgt_full] = val

    for tgt_full in range(config.N_POP_UNITS):
        if tgt_full < 4:
            full[config.N_POP_UNITS:, tgt_full] = \
                sub_vars[tgt_full][var_name][rec_end_local:inp_end_local, 0]
        elif tgt_full == 4:
            full[config.N_POP_UNITS:, 4] = np.mean(
                [sub_vars[X][var_name][rec_end_local:inp_end_local, 1]
                 for X in range(4)], axis=0)
        elif tgt_full == 5:
            full[config.N_POP_UNITS:, 5] = np.mean(
                [sub_vars[X][var_name][rec_end_local:inp_end_local, 2]
                 for X in range(4)], axis=0)
    return full


def _lift_per_unit_var(sub_vars, var_name):
    """Lift a per-unit variable (I_ext) from 4 sub-models."""
    full = np.zeros(config.N_POP_UNITS, dtype=np.float32)
    for X in range(4):
        full[X] = sub_vars[X][var_name][0]
    full[4] = float(np.mean([sub_vars[X][var_name][1] for X in range(4)]))
    full[5] = float(np.mean([sub_vars[X][var_name][2] for X in range(4)]))
    return full


def lift_phase1_to_full(phase1_dir, verbose=True):
    """Build a fresh WC full model and assign lifted values to its variables.

    The lifted physical values (gsyn_max, tau_1, tau_2) are converted back
    to the cell's theta parameters via inverse softplus / log, so the
    effective synapse values match exactly what each sub-model produced.
    """
    sub_vars = _load_submodel_vars(phase1_dir)
    if verbose:
        print(f"  Loaded sub-model vars from {phase1_dir} for X_idx 0..3.")

    model = build_model(lr=config.LEARNING_RATE)
    layer = model.get_layer("wc_rnn").cell

    gsyn_max_full = _lift_matrix_var(sub_vars, "gsyn_max")
    tau_1_full = _lift_matrix_var(sub_vars, "tau_1")
    tau_2_full = _lift_matrix_var(sub_vars, "tau_2")
    I_ext_full = _lift_per_unit_var(sub_vars, "I_ext")

    theta_gsyn_full = _inv_softplus_np(
        gsyn_max_full * config.SYN_GSYN_INIT_SCALE)
    theta_tau_1_full = np.log(np.maximum(tau_1_full, 1e-7))
    theta_tau_2_full = np.log(np.maximum(tau_2_full, 1e-7))

    lifts = {
        "I_ext": I_ext_full,
        "_theta_gsyn": theta_gsyn_full,
        "_theta_tau_1": theta_tau_1_full,
        "_theta_tau_2": theta_tau_2_full,
    }
    for var_name, arr in lifts.items():
        var = getattr(layer, var_name)
        assert tuple(var.shape) == arr.shape, (
            f"{var_name}: var.shape={tuple(var.shape)} "
            f"vs lifted.shape={arr.shape}")
        var.assign(arr)

    if verbose:
        print(f"  Lifted variable shapes:")
        for k, arr in lifts.items():
            print(f"    {k}: {arr.shape} "
                  f"(min={arr.min():.4f}, max={arr.max():.4f}, "
                  f"mean={arr.mean():.4f})")

    return model, {
        "I_ext": I_ext_full,
        "gsyn_max": gsyn_max_full,
        "tau_1": tau_1_full,
        "tau_2": tau_2_full,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1-dir", type=str,
                   default=os.path.join(config.RESULTS_DIR, "phase1_wc"))
    p.add_argument("--out", type=str, default=None,
                   help="Output h5 path (default: "
                        "<phase1-dir>/lifted.weights.h5).")
    p.add_argument("--no-save-npz", action="store_true",
                   help="Skip saving lifted.weights.npz mirror.")
    args = p.parse_args()

    print("Configuring devices...")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    out_path = args.out or os.path.join(args.phase1_dir,
                                        "lifted.weights.h5")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Lifting sub-models from {args.phase1_dir}...")
    model, lifted = lift_phase1_to_full(args.phase1_dir)
    n_vars = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"  Full-model trainable parameters: {n_vars}")

    model.save_weights(out_path)
    print(f"  Saved lifted weights -> {out_path}")

    if not args.no_save_npz:
        npz_path = os.path.splitext(out_path)[0] + ".npz"
        np.savez(
            npz_path,
            I_ext=lifted["I_ext"],
            gsyn_max=lifted["gsyn_max"],
            tau_1=lifted["tau_1"],
            tau_2=lifted["tau_2"],
        )
        print(f"  Saved lifted npz mirror -> {npz_path}")

    summary_path = os.path.join(args.phase1_dir, "phase2_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "phase1_dir": args.phase1_dir,
            "out_path": out_path,
            "n_trainable": n_vars,
        }, f, indent=2)

    print(f"\nPhase 2 (WC) complete.")
    print(f"Resume full training with:")
    print(f"  python train_wc_nonpsyns.py --resume {out_path}")


if __name__ == "__main__":
    main()