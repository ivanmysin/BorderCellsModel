"""Phase 2: lift 4 sub-models into the full 6-unit model and continue training.

Reads weights from ``results/phase1/{border_N,border_S,border_E,border_W}_submodel_vars.npz``
(produced by ``train_phase1.py``), assembles them into the full-model
variable tensors, then trains the full stateful RNN starting from the lifted
weights. Same MSE + decorrelation penalty loss as ``train_simple.py``.

Lifting strategy (defaults):
- pyramidal self-recurrent Border_X → Border_X: from sub-model X's recurrent.
- inter-pyramidal Border_Y → Border_X (incoming to X): from sub-model X's
  teacher matrix at the row for Y (sub-model X trained Border_X with the
  other pyramids as teacher inputs).
- inter-pyramidal Border_X → Border_Y (outgoing from X): from sub-model Y's
  teacher matrix at the row for X (sub-model Y trained Border_Y with Border_X
  as one of its teacher inputs).
- Border_X ↔ Basket/Axo (both directions): from sub-model X's recurrent.
- Basket ↔ Axo self & mutual, real-input synapses to Basket/Axo:
  averaged across the 4 sub-models.
- I_ext: per-Border uses its own sub-model; Basket/Axo use the average.

Usage:
    python train_phase2.py [--phase1-dir DIR] [--epochs 100] [--lr 1e-3] [--seed 42]
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import config
try:
    from train_simple import (
        BorderMeanFieldNetwork, build_model, setup_gpu, decorrelation_penalty,
    )
except ModuleNotFoundError as e:
    sys.stderr.write(
        f"ERROR: {e}\n"
        f"  train_phase2.py expects train_simple.py next to it.\n"
    )
    raise
try:
    from utils.dataset import load_dataset_hdf5
except ModuleNotFoundError as e:
    sys.stderr.write(
        f"ERROR: {e}\n"
        f"  train_phase2.py expects utils/dataset.py next to it.\n"
    )
    raise


PYRAMIDAL_NAMES = ["border_N", "border_S", "border_E", "border_W"]


def _load_submodel_vars(phase1_dir):
    """Load 4 sub-model vars npz. Returns dict {X_idx: {var_name: array}}."""
    out = {}
    for X_idx, name in enumerate(PYRAMIDAL_NAMES):
        path = os.path.join(phase1_dir, f"{name}_submodel_vars.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run train_phase1.py first.")
        data = np.load(path)
        out[X_idx] = {k: data[k] for k in data.files}
    return out


def _lift_matrix_var(sub_vars, var_name):
    """Lift a (pre, post) shape synapse param from 4 sub-models to full model.

    Sub-model pre layout:    [recurrent(3); real_inputs(21); teacher(3)]
    Sub-model post layout:   [Border_X, Basket, Axo]
    Full-model pre layout:   [recurrent(6); real_inputs(21)]
    Full-model post layout:  [Border_N, Border_S, Border_E, Border_W, Basket, Axo]
    """
    full_pre  = config.N_POP_UNITS + config.N_INPUTS        # 27
    full_post = config.N_POP_UNITS                          # 6
    full = np.zeros((full_pre, full_post), dtype=np.float32)

    rec_end_local    = 3                                    # sub-model recurrent rows
    inp_end_local    = 3 + config.N_INPUTS                  # 24
    teacher_off_local = inp_end_local                       # 24

    for src_full in range(config.N_POP_UNITS):              # rows 0..5
        for tgt_full in range(config.N_POP_UNITS):          # cols 0..5
            if src_full < 4 and tgt_full < 4:
                if src_full == tgt_full:
                    val = sub_vars[tgt_full][var_name][0, 0]
                else:
                    other_tgt = [i for i in range(4) if i != tgt_full]
                    teacher_row = other_tgt.index(src_full)
                    val = sub_vars[tgt_full][var_name][teacher_off_local + teacher_row, 0]
            elif tgt_full < 4 and src_full in (4, 5):
                val = sub_vars[tgt_full][var_name][src_full - 3, 0]
            elif tgt_full == 4 and src_full < 4:
                val = sub_vars[src_full][var_name][0, 1]
            elif tgt_full == 4 and src_full == 4:
                val = float(np.mean([sub_vars[X][var_name][1, 1] for X in range(4)]))
            elif tgt_full == 4 and src_full == 5:
                val = float(np.mean([sub_vars[X][var_name][2, 1] for X in range(4)]))
            elif tgt_full == 5 and src_full < 4:
                val = sub_vars[src_full][var_name][0, 2]
            elif tgt_full == 5 and src_full == 4:
                val = float(np.mean([sub_vars[X][var_name][1, 2] for X in range(4)]))
            elif tgt_full == 5 and src_full == 5:
                val = float(np.mean([sub_vars[X][var_name][2, 2] for X in range(4)]))
            else:
                raise RuntimeError(f"unhandled src={src_full}, tgt={tgt_full}")
            full[src_full, tgt_full] = val

    for tgt_full in range(config.N_POP_UNITS):
        if tgt_full < 4:
            full[config.N_POP_UNITS:, tgt_full] = \
                sub_vars[tgt_full][var_name][rec_end_local:inp_end_local, 0]
        elif tgt_full == 4:
            full[config.N_POP_UNITS:, 4] = np.mean(
                [sub_vars[X][var_name][rec_end_local:inp_end_local, 1] for X in range(4)],
                axis=0)
        elif tgt_full == 5:
            full[config.N_POP_UNITS:, 5] = np.mean(
                [sub_vars[X][var_name][rec_end_local:inp_end_local, 2] for X in range(4)],
                axis=0)
    return full


def _lift_per_unit_var(sub_vars, var_name):
    """Lift a per-unit variable (I_ext) from 4 sub-models."""
    full = np.zeros(config.N_POP_UNITS, dtype=np.float32)
    for X in range(4):
        full[X] = sub_vars[X][var_name][0]
    full[4] = float(np.mean([sub_vars[X][var_name][1] for X in range(4)]))
    full[5] = float(np.mean([sub_vars[X][var_name][2] for X in range(4)]))
    return full


def lift_phase1_to_full(phase1_dir):
    """Build a fresh full model and assign lifted values to its variables."""
    sub_vars = _load_submodel_vars(phase1_dir)
    print(f"  Loaded sub-model vars from {phase1_dir} for X_idx 0..3.")

    model = build_model(lr=config.LEARNING_RATE)
    layer = model.get_layer("border_rnn").cell

    lifts = {
        "I_ext":    _lift_per_unit_var(sub_vars, "I_ext"),
        "gsyn_max": _lift_matrix_var(sub_vars, "gsyn_max"),
        "tau_f":    _lift_matrix_var(sub_vars, "tau_f"),
        "tau_d":    _lift_matrix_var(sub_vars, "tau_d"),
        "tau_r":    _lift_matrix_var(sub_vars, "tau_r"),
        "Uinc":     _lift_matrix_var(sub_vars, "Uinc"),
    }

    for var_name, arr in lifts.items():
        var = getattr(layer, var_name)
        assert tuple(var.shape) == arr.shape, (
            f"{var_name}: var.shape={tuple(var.shape)} vs lifted.shape={arr.shape}")
        var.assign(arr)

    print(f"  Lifted variable shapes:")
    for k, arr in lifts.items():
        print(f"    {k}: {arr.shape}")

    return model


def _load_all_batches(dataset_path):
    ds = load_dataset_hdf5(dataset_path)
    n = ds["n_batches"]
    Xl, Yl = [], []
    for i in range(n):
        b = ds["get_batch"](i)
        Xl.append(b["inputs"])
        Yl.append(b["targets"])
    ds["file"].close()
    return np.concat(Xl).astype(np.float32), np.concat(Yl).astype(np.float32)


def _mse_loss_with_reg(y_true, y_pred):
    k = tf.shape(y_true)[-1]
    pred_subset = y_pred[..., :k]
    mse = tf.reduce_mean(tf.square(y_true - pred_subset))
    return mse + config.WTA_WEIGHT * decorrelation_penalty(pred_subset)


def _train_stateful(model, X, Y, n_batches, n_epochs, batches_per_epoch,
                    start_batch, log_every, checkpoint_dir):
    """Train via `model.fit()` with one long sequence per epoch.

    Each epoch concatenates `batches_per_epoch` consecutive stored batches
    into a single (1, K*T, n_inputs) sequence. The stateful RNN processes
    the long sequence in one `fit(epochs=1)` call, so state propagates
    across the K batches. `reset_states()` clears it at epoch boundaries.
    """
    rnn = model.get_layer("border_rnn")
    max_start = max(0, n_batches - batches_per_epoch)
    T = X.shape[1]
    n_inputs = X.shape[2]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(model.optimizer.learning_rate),
            clipvalue=10.0),
        loss=_mse_loss_with_reg,
    )

    loss_history = []
    t_start = time.time()
    for epoch in range(n_epochs):
        rnn.reset_states()
        s = start_batch if start_batch is not None else (
            int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0)

        x = X[s:s + batches_per_epoch]
        y = Y[s:s + batches_per_epoch]
        x = np.ascontiguousarray(x.reshape(1, batches_per_epoch * T, n_inputs))
        y = np.ascontiguousarray(y.reshape(1, batches_per_epoch * T, y.shape[-1]))

        hist = model.fit(x, y, epochs=1, verbose=0,
                         batch_size=1, shuffle=False)
        loss = float(hist.history['loss'][0])
        if not np.isfinite(loss):
            print(f"\n  NaN/Inf at epoch {epoch+1} "
                  f"(loss={loss}); resetting state and skipping.")
            rnn.reset_states()
            continue
        loss_history.append(loss)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"  epoch {epoch+1:4d}/{n_epochs} | loss={loss:.6f} "
                  f"| elapsed={(time.time() - t_start)/60:.1f} min "
                  f"| start_batch={s}")

        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            tag = f"phase2_epoch_{epoch+1:04d}_loss_{loss:.6f}"
            weights_path = os.path.join(checkpoint_dir, f"{tag}.weights.h5")
            model.save_weights(weights_path)
            with open(os.path.join(checkpoint_dir, f"{tag}_meta.json"), "w") as f:
                json.dump({"epoch": epoch + 1, "loss": loss}, f, indent=2)
    return loss_history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1-dir", type=str,
                   default=os.path.join(config.RESULTS_DIR, "phase1"))
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batches-per-epoch", type=int, default=None)
    p.add_argument("--start-batch", type=int, default=0)
    p.add_argument("--checkpoint-dir", type=str,
                   default=os.path.join(config.RESULTS_DIR, "phase2"))
    args = p.parse_args()

    n_epochs = args.epochs or config.N_EPOCHS
    lr = args.lr or config.LEARNING_RATE
    seed = args.seed if args.seed is not None else config.RANDOM_SEED
    batches_per_epoch = args.batches_per_epoch or config.N_BATCHES_PER_EPOCH

    print("Configuring devices...")
    setup_gpu()

    np.random.seed(seed)
    tf.random.set_seed(seed)

    print(f"Lifting sub-models from {args.phase1_dir}...")
    model = lift_phase1_to_full(args.phase1_dir)
    n_vars = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"  Full-model trainable parameters: {n_vars}")

    ds_path = args.dataset or os.path.join(
        os.path.dirname(config.TRAJECTORY_HDF5), "dataset.h5")
    print(f"Loading dataset from {ds_path}...")
    X, Y = _load_all_batches(ds_path)
    print(f"  X={X.shape}, Y={Y.shape}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model.optimizer.learning_rate.assign(lr)
    n_batches = X.shape[0]
    log_every = max(1, n_epochs // 20)
    hist = _train_stateful(
        model, X, Y, n_batches, n_epochs, batches_per_epoch,
        args.start_batch, log_every, args.checkpoint_dir,
    )

    final_path = os.path.join(args.checkpoint_dir, "latest.weights.h5")
    model.save_weights(final_path)
    np.savez(os.path.join(args.checkpoint_dir, "phase2_loss_history.npz"),
             loss_history=np.asarray(hist, dtype=np.float64))
    print(f"\nPhase 2 complete. Final weights → {final_path}")


if __name__ == "__main__":
    main()
