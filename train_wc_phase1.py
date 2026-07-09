"""Phase 1 (WC): train 4 Wilson-Cowan sub-models, one per Border cell.

Each sub-model has 3 units (Border_X, Basket, Axo) and 24 input channels:
21 real (d_far, d_near, speed, HDx18) + 3 teacher channels carrying the
ideal targets of the OTHER three pyramids. Borders are predicted one at
a time; stateful RNN training (state propagates across batches within an
epoch).

Saves trained weights per sub-model to
``results/phase1_wc/border_{N,S,E,W}_submodel.weights.h5`` and a matching
``.npz`` with the same variable tensors for fast lifting in Phase 2.

Usage:
    python train_wc_phase1.py [--epochs 30] [--lr 1e-3] [--seed 42]
                              [--teacher-gsyn 1.0] [--checkpoint-dir DIR]
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
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import RNN
from tensorflow.keras.optimizers import Adam

import config
try:
    from utils.submodel_wc import (
        WilsonCowanSubNetwork,
        build_submodel_wc_params,
        augment_with_teachers,
        extract_target_for,
    )
except ModuleNotFoundError as e:
    sys.stderr.write(
        f"ERROR: {e}\n"
        f"  train_wc_phase1.py expects utils/submodel_wc.py next to it.\n"
        f"  Make sure utils/submodel_wc.py exists in: "
        f"{PROJECT_ROOT / 'utils' / 'submodel_wc.py'}\n"
    )
    raise

try:
    from utils.dataset import load_dataset_hdf5
except ModuleNotFoundError as e:
    sys.stderr.write(
        f"ERROR: {e}\n"
        f"  train_wc_phase1.py expects utils/dataset.py next to it.\n"
    )
    raise

tf.get_logger().setLevel("ERROR")

PYRAMIDAL_NAMES = ["border_N", "border_S", "border_E", "border_W"]


def _mse_loss(y_true, y_pred):
    k = tf.shape(y_true)[-1]
    pred_subset = y_pred[..., :k]
    return tf.reduce_mean(tf.square(y_true - pred_subset))


def _build_submodel(params, lr, dt):
    n_pre = int(params["pconn"].shape[0])
    n_post = int(params["pconn"].shape[1])
    n_inputs = n_pre - n_post
    inputs = Input(shape=(None, n_inputs), batch_size=1)
    cell = WilsonCowanSubNetwork(params, dt=dt, batch_size=1,
                                 n_pre=n_pre, n_units=n_post, n_post=n_post)
    rnn = RNN(cell, return_sequences=True, stateful=True,
              name="wc_rnn_sub")
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


def _train_stateful(model, X, Y_target, n_batches, n_epochs,
                    batches_per_epoch, start_batch, log_every,
                    checkpoint_dir, tag_prefix):
    """Train via a single `model.fit(X, Y, epochs=n_epochs)` call.

    Concatenates `batches_per_epoch` consecutive stored batches into one
    (1, K*T, n_inputs) sequence. The same data is used for `n_epochs`
    Keras epochs. A callback resets the stateful RNN's state at every
    epoch boundary, so state propagates within the K*T-step window but
    starts fresh each epoch.
    """
    rnn_layer = model.get_layer("wc_rnn_sub")
    max_start = max(0, n_batches - batches_per_epoch)
    T = X.shape[1]
    n_inputs = X.shape[2]
    n_targets = Y_target.shape[2]

    if start_batch is None:
        s = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
    else:
        s = start_batch

    x = X[s:s + batches_per_epoch]
    y = Y_target[s:s + batches_per_epoch]
    x = np.ascontiguousarray(x.reshape(1, batches_per_epoch * T, n_inputs))
    y = np.ascontiguousarray(y.reshape(1, batches_per_epoch * T, n_targets))

    loss_history = []

    class EpochCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            rnn_layer.reset_states()

        def on_epoch_end(self, epoch, logs=None):
            loss = float(logs.get('loss'))
            loss_history.append(loss)
            if not np.isfinite(loss):
                print(f"\n  NaN/Inf at epoch {epoch+1} (loss={loss}).")
            if (epoch + 1) % log_every == 0 or epoch == 0:
                print(f"  epoch {epoch+1:4d}/{n_epochs} | loss={loss:.6f} "
                      f"| start_batch={s}")
            if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
                tag = f"{tag_prefix}_epoch_{epoch+1:04d}_loss_{loss:.6f}"
                model.save_weights(
                    os.path.join(checkpoint_dir, f"{tag}.weights.h5"))
                with open(os.path.join(checkpoint_dir, f"{tag}_meta.json"),
                          "w") as f:
                    json.dump({"epoch": epoch + 1, "loss": loss,
                               "tag_prefix": tag_prefix}, f, indent=2)

    model.fit(x, y, epochs=n_epochs, batch_size=1, shuffle=False,
              verbose=0, callbacks=[EpochCallback()])
    return loss_history


def train_one_submodel(X_idx, X_aug, Y_target, n_epochs, lr,
                       batches_per_epoch, start_batch, checkpoint_dir,
                       log_every, seed, teacher_gsyn, dt):
    name = PYRAMIDAL_NAMES[X_idx]

    np.random.seed(seed + X_idx)
    tf.random.set_seed(seed + X_idx)

    print(f"\n=== Sub-model for {name} (X_idx={X_idx}) ===")
    print(f"  Input shape: {X_aug.shape}, target shape: {Y_target.shape}")

    params = build_submodel_wc_params(X_idx, teacher_gsyn=teacher_gsyn,
                                      rng_seed=seed)
    model = _build_submodel(params, lr=lr, dt=dt)
    n_vars = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"  Trainable parameters: {n_vars} "
          f"({len(model.trainable_variables)} vars)")
    print(f"  Trainable variables: "
          f"{[v.name for v in model.trainable_variables]}")

    n_batches = X_aug.shape[0]
    tag_prefix = f"submodel_{name}"
    hist = _train_stateful(
        model, X_aug, Y_target, n_batches, n_epochs, batches_per_epoch,
        start_batch, log_every, checkpoint_dir, tag_prefix,
    )

    final_path = os.path.join(checkpoint_dir, f"{name}_submodel.weights.h5")
    model.save_weights(final_path)
    print(f"  Final weights saved to {final_path}")

    layer = model.get_layer("wc_rnn_sub").cell
    npz_path = os.path.join(checkpoint_dir, f"{name}_submodel_vars.npz")
    np.savez(
        npz_path,
        I_ext=layer.I_ext.numpy(),
        theta_gsyn=layer._theta_gsyn.numpy(),
        theta_tau_1=layer._theta_tau_1.numpy(),
        theta_tau_2=layer._theta_tau_2.numpy(),
        gsyn_max=layer._get_gsyn().numpy(),
        tau_1=layer._get_tau_1().numpy(),
        tau_2=layer._get_tau_2().numpy(),
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
    p.add_argument("--teacher-gsyn", type=float, default=1.0,
                   help="Initial gsyn_max on teacher rows "
                        "(default 1.0, vs ~1e-4 random).")
    p.add_argument("--checkpoint-dir", type=str,
                   default=os.path.join(config.RESULTS_DIR, "phase1_wc"))
    args = p.parse_args()

    n_epochs = args.epochs or config.N_EPOCHS
    lr = args.lr or config.LEARNING_RATE
    seed = args.seed if args.seed is not None else config.RANDOM_SEED
    batches_per_epoch = (args.batches_per_epoch
                         or config.N_BATCHES_PER_EPOCH)

    print("Configuring devices...")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

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
            teacher_gsyn=args.teacher_gsyn,
            dt=config.DT,
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
            "teacher_gsyn": args.teacher_gsyn,
            "saved_weights": saved_paths,
        }, f, indent=2)
    print(f"\nPhase 1 (WC) complete. Summary -> {summary_path}")


if __name__ == "__main__":
    main()