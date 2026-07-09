"""Phase 1: train 4 sub-models, one per Border cell.

Each sub-model has 3 units (Border_X, Basket, Axo) and 24 input channels:
21 real (d_far, d_near, speed, HD×18) + 3 teacher channels carrying the ideal
targets of the OTHER three pyramids. Borders are predicted one-at-a-time;
stateful RNN training (state propagates across batches within an epoch).

Saves trained weights per sub-model to ``results/phase1/border_{N,S,E,W}.weights.h5``.

Usage:
    python train_phase1.py [--epochs 30] [--lr 5e-3] [--seed 42]
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import RNN
from tensorflow.keras.optimizers import Adam

import config
from train_simple import (
    BorderMeanFieldNetwork, MinMax, setup_gpu,
)
from utils.submodel import (
    build_submodel_params, augment_with_teachers, extract_target_for,
)
from utils.dataset import load_dataset_hdf5


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

PYRAMIDAL_NAMES = ["border_N", "border_S", "border_E", "border_W"]


def _mse_loss(y_true, y_pred):
    """Pure MSE on the first ``k`` units of y_pred, where k = #target channels."""
    k = tf.shape(y_true)[-1]
    pred_subset = y_pred[..., :k]
    return tf.reduce_mean(tf.square(y_true - pred_subset))


def _build_submodel(params, lr):
    n_inputs = int(params["pconn"].shape[0]) - int(params["alpha"].shape[0])
    inputs = Input(shape=(None, n_inputs), batch_size=1)
    cell = BorderMeanFieldNetwork(params, dt_dim=config.DT, batch_size=1)
    rnn = RNN(cell, return_sequences=True, stateful=True, name="border_rnn_sub")
    out = rnn(inputs)
    model = Model(inputs, out)
    model.compile(
        optimizer=Adam(learning_rate=lr, clipvalue=10.0),
        loss=_mse_loss,
    )
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


def _train_stateful(model, X, Y_target, n_batches, n_epochs, batches_per_epoch,
                    start_batch, log_every, checkpoint_dir, tag_prefix):
    rnn_layer = model.get_layer("border_rnn_sub")
    rnn_layer.reset_states()
    max_start = max(0, n_batches - batches_per_epoch)
    loss_history = []
    t_start = time.time()
    for epoch in range(n_epochs):
        rnn_layer.reset_states()
        s = start_batch if start_batch is not None else (
            int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0)

        epoch_loss = 0.0
        n_finite = 0
        for i in range(batches_per_epoch):
            x = X[s + i:s + i + 1]
            y = Y_target[s + i:s + i + 1]
            loss = model.train_on_batch(x, y)
            v = float(loss)
            if not np.isfinite(v):
                print(f"\n  NaN/Inf at epoch {epoch+1}, batch {i+1} "
                      f"(loss={v}); resetting state and skipping to next epoch.")
                rnn_layer.reset_states()
                continue
            epoch_loss += v
            n_finite += 1
        avg = epoch_loss / max(n_finite, 1)
        loss_history.append(avg)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"  epoch {epoch+1:4d}/{n_epochs} | loss={avg:.6f} "
                  f"| elapsed={(time.time() - t_start)/60:.1f} min "
                  f"| start_batch={s}")

        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            tag = f"{tag_prefix}_epoch_{epoch+1:04d}_loss_{avg:.6f}"
            model.save_weights(os.path.join(checkpoint_dir, f"{tag}.weights.h5"))
            with open(os.path.join(checkpoint_dir, f"{tag}_meta.json"), "w") as f:
                json.dump({"epoch": epoch + 1, "loss": avg,
                           "tag_prefix": tag_prefix}, f, indent=2)
    return loss_history


def train_one_submodel(X_idx: int, X_aug: np.ndarray, Y_target: np.ndarray,
                       n_epochs: int, lr: float, batches_per_epoch: int,
                       start_batch, checkpoint_dir: str,
                       log_every: int, seed: int):
    """Train one 3-unit sub-model for Border_X_idx. Returns path to weights."""
    name = PYRAMIDAL_NAMES[X_idx]

    np.random.seed(seed + X_idx)
    tf.random.set_seed(seed + X_idx)

    print(f"\n=== Sub-model for {name} (X_idx={X_idx}) ===")
    print(f"  Input shape: {X_aug.shape}, target shape: {Y_target.shape}")

    params = build_submodel_params(X_idx)
    model = _build_submodel(params, lr=lr)
    n_vars = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"  Trainable parameters: {n_vars}  ({len(model.trainable_variables)} vars)")
    print(f"  Trainable variables: {[v.name for v in model.trainable_variables]}")

    n_batches = X_aug.shape[0]
    tag_prefix = f"submodel_{name}"
    hist = _train_stateful(
        model, X_aug, Y_target, n_batches, n_epochs, batches_per_epoch,
        start_batch, log_every, checkpoint_dir, tag_prefix,
    )

    final_path = os.path.join(checkpoint_dir, f"{name}_submodel.weights.h5")
    model.save_weights(final_path)
    print(f"  Final weights saved to {final_path}")

    layer = model.get_layer("border_rnn_sub").cell
    var_names = ["I_ext", "gsyn_max", "tau_f", "tau_d", "tau_r", "Uinc"]
    npz_path = os.path.join(checkpoint_dir, f"{name}_submodel_vars.npz")
    np.savez(
        npz_path,
        **{name: getattr(layer, name).numpy() for name in var_names},
    )
    print(f"  Variable npz saved to {npz_path}")
    return final_path, hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batches-per-epoch", type=int, default=None)
    p.add_argument("--start-batch", type=int, default=0,
                   help="Fix starting stored batch (default 0).")
    p.add_argument("--checkpoint-dir", type=str,
                   default=os.path.join(config.RESULTS_DIR, "phase1"))
    args = p.parse_args()

    n_epochs   = args.epochs   or config.N_EPOCHS
    lr         = args.lr       or config.LEARNING_RATE
    seed       = args.seed     if args.seed is not None else config.RANDOM_SEED
    batches_per_epoch = args.batches_per_epoch or config.N_BATCHES_PER_EPOCH

    print("Configuring devices...")
    setup_gpu()

    ds_path = args.dataset or os.path.join(
        os.path.dirname(config.TRAJECTORY_HDF5), "dataset.h5")
    print(f"Loading dataset from {ds_path}...")
    X, Y = _load_all_batches(ds_path)
    print(f"  X={X.shape}, Y={Y.shape}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    saved_paths = []
    for X_idx in range(4):
        X_aug = augment_with_teachers(X, Y, X_idx)
        Y_target = extract_target_for(Y, X_idx)
        path, hist = train_one_submodel(
            X_idx, X_aug, Y_target,
            n_epochs=n_epochs, lr=lr,
            batches_per_epoch=batches_per_epoch,
            start_batch=args.start_batch,
            checkpoint_dir=args.checkpoint_dir,
            log_every=max(1, n_epochs // 20),
            seed=seed,
        )
        saved_paths.append(path)
        np.savez(
            os.path.join(args.checkpoint_dir,
                         f"{PYRAMIDAL_NAMES[X_idx]}_loss_history.npz"),
            loss_history=np.asarray(hist, dtype=np.float64),
        )

    summary_path = os.path.join(args.checkpoint_dir, "phase1_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "n_epochs": n_epochs,
            "lr": lr,
            "batches_per_epoch": batches_per_epoch,
            "seed": seed,
            "saved_weights": saved_paths,
        }, f, indent=2)
    print(f"\nPhase 1 complete. Summary → {summary_path}")


if __name__ == "__main__":
    main()
