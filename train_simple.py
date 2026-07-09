"""Train border cell network via a single Keras Layer (no neuraltide).

One Layer wraps the whole system:
  - Mean-field Izhikevich (r, v, w) — RK4 with /tau_pop
  - Tsodyks-Markram synapses (R, U, A) — analytic single-step
Wrapped in RNN(..., stateful=True); custom training loop with
model.train_on_batch. State propagates across the whole run (no reset).

Usage:
    python train_simple.py [--dataset data/dataset.h5] [--epochs 100]
                           [--lr 1e-3] [--seed 42] [--batches-per-epoch 50]
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import RNN, Layer
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.optimizers import Adam
import h5py

import config
from utils.params import (
    build_pop_params,
    build_inp_gsyn_matrix, build_inp_tau_f_matrix, build_inp_tau_d_matrix,
    build_inp_tau_r_matrix, build_inp_Uinc_matrix, build_inp_pconn_matrix,
    build_inp_e_r_matrix,
    build_rec_gsyn_matrix, build_rec_tau_f_matrix, build_rec_tau_d_matrix,
    build_rec_tau_r_matrix, build_rec_Uinc_matrix, build_rec_pconn_matrix,
    build_rec_e_r_matrix,
)
from utils.dataset import load_dataset_hdf5


class NaNStoppingCallback(tf.keras.callbacks.Callback):
    """Stop training if loss becomes NaN/Inf."""

    def on_batch_end(self, batch, logs=None):
        loss = logs.get('loss')
        if loss is None or not np.isfinite(loss):
            print(f"\n  NaN/Inf detected at batch {batch} (loss={loss}), stopping.")
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        if loss is None or not np.isfinite(loss):
            print(f"\n  NaN/Inf detected at epoch {epoch} (loss={loss}), stopping.")
            self.model.stop_training = True


class CheckpointCallback(tf.keras.callbacks.Callback):
    """Save weights + loss history at each epoch with unique filenames.

    Files saved per epoch:
        results/checkpoints/epoch_{N:04d}_loss_{L:.6f}.weights.h5
        results/checkpoints/epoch_{N:04d}_loss_{L:.6f}_meta.json
    """

    def __init__(self, checkpoint_dir=None):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            config.RESULTS_DIR, 'checkpoints')
        self.loss_history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = float(logs.get('loss', float('nan')))
        self.loss_history.append(loss)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        tag = f"epoch_{epoch + 1:04d}_loss_{loss:.6f}"

        weights_path = os.path.join(self.checkpoint_dir, f"{tag}.weights.h5")
        self.model.save_weights(weights_path)

        meta = {
            'epoch': epoch + 1,
            'loss': loss,
            'val_loss': logs.get('val_loss'),
            'learning_rate': float(self.model.optimizer.learning_rate.numpy()),
        }
        meta_path = os.path.join(self.checkpoint_dir, f"{tag}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        if (epoch + 1) % max(1, self.params.get('epochs', 100) // 20) == 0 or epoch == 0:
            print(f"  [checkpoint] epoch {epoch + 1}: loss={loss:.6f} → {weights_path}")

    def on_train_end(self, logs=None):
        latest_path = os.path.join(self.checkpoint_dir, 'latest.weights.h5')
        self.model.save_weights(latest_path)
        history_path = os.path.join(self.checkpoint_dir, 'loss_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.loss_history, f, indent=2)
        print(f"  [checkpoint] saved latest weights + loss_history.json")


class MinMax(Constraint):
    def __init__(self, min_val=0.0, max_val=float('inf')):
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_val, self.max_val)

    def get_config(self):
        return {'min_val': self.min_val, 'max_val': self.max_val}


class BorderMeanFieldNetwork(Layer):
    """Single-layer mean-field Izhikevich + Tsodyks-Markram.

    State: r, v, w [1, units] + R, U, A [1, pre, post]
    Trainable: gsyn_max, I_ext (others frozen by config.TRAIN_* flags).
    """

    def __init__(self, params, dt_dim=0.1, batch_size=1, **kwargs):
        super().__init__(**kwargs)
        self.dt_dim = float(dt_dim)
        self.units = int(np.asarray(params['alpha']).shape[0])
        self.pre = int(np.asarray(params['pconn']).shape[0])
        self.post = int(np.asarray(params['pconn']).shape[1])
        self.PI = float(np.pi)
        self.v_max = 10.0

        self.alpha = tf.constant(params['alpha'], dtype=tf.float32)
        self.a = tf.constant(params['a'], dtype=tf.float32)
        self.b = tf.constant(params['b'], dtype=tf.float32)
        self.w_jump = tf.constant(params['w_jump'], dtype=tf.float32)
        self.tau_pop = tf.constant(params['tau_pop'], dtype=tf.float32)
        self.pconn = tf.constant(params['pconn'], dtype=tf.float32)
        self.e_r = tf.constant(params['e_r'], dtype=tf.float32)

        # print("=== DEBUG ===")
        # print(type(self.alpha))
        # print(type(self.e_r))

        self.Delta_I = self.add_weight(
            shape=(self.units,),
            initializer=tf.constant_initializer(params['Delta_I']),
            trainable=config.TRAIN_POP_DELTA_I,
            constraint=MinMax(0.0001, 0.1),
            name='Delta_I',
        )
        self.I_ext = self.add_weight(
            shape=(self.units,),
            initializer=tf.constant_initializer(params['I_ext']),
            trainable=config.TRAIN_POP_IEXT,
            constraint=MinMax(-0.5, 0.5),
            name='I_ext',
        )
        self.gsyn_max = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(params['gsyn_max']),
            trainable=config.TRAIN_SYNAPSE_GMAX,
            constraint=tf.keras.constraints.NonNeg(),
            regularizer=tf.keras.regularizers.l2(config.L2_GSYN_WEIGHT),
            name='gsyn_max',
        )
        self.tau_f = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(params['tau_f']),
            trainable=config.TRAIN_SYNAPSE_TAU_f,
            constraint=MinMax(6.0, 240.0),
            name='tau_f',
        )
        self.tau_d = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(params['tau_d']),
            trainable=config.TRAIN_SYNAPSE_TAU_d,
            constraint=MinMax(2.0, 15.0),
            name='tau_d',
        )
        self.tau_r = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(params['tau_r']),
            trainable=config.TRAIN_SYNAPSE_TAU_r,
            constraint=MinMax(91.0, 1300.0),
            name='tau_r',
        )
        self.Uinc = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(params['Uinc']),
            trainable=config.TRAIN_SYNAPSE_U,
            constraint=MinMax(0.04, 0.7),
            name='Uinc',
        )

        self.state_size = [
            tf.TensorShape([batch_size, self.units]),
            tf.TensorShape([batch_size, self.units]),
            tf.TensorShape([batch_size, self.units]),
            tf.TensorShape([batch_size, self.pre, self.post]),
            tf.TensorShape([batch_size, self.pre, self.post]),
            tf.TensorShape([batch_size, self.pre, self.post]),
        ]
        self.output_size = self.units

    def get_initial_state(self, batch_size=1):
        return [
            tf.random.uniform([batch_size, self.units], minval=0.0,
                              maxval=0.1, dtype=tf.float32),
            tf.random.normal([batch_size, self.units], mean=0.0,
                             stddev=0.01, dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.ones([batch_size, self.pre, self.post], dtype=tf.float32),
            tf.zeros([batch_size, self.pre, self.post], dtype=tf.float32),
            tf.zeros([batch_size, self.pre, self.post], dtype=tf.float32),
        ]

    def _deriv(self, r, v, w, A):
        g_syn = self.gsyn_max * A
        g_syn_tot = tf.reduce_sum(g_syn, axis=1)
        I_syn = tf.reduce_sum(g_syn * (self.e_r[tf.newaxis, :, :] - v[:, tf.newaxis, :]), axis=1)
        drdt = (self.Delta_I / self.PI
                + 2.0 * r * v
                - (self.alpha + g_syn_tot) * r) / self.tau_pop
        v_sq = v * v / (1.0 + (v / self.v_max) ** 2)
        dvdt = (v_sq
                - self.alpha * v
                - w
                + self.I_ext
                + I_syn
                - (self.PI * r) ** 2) / self.tau_pop
        dwdt = (self.a * (self.b * v - w) + self.w_jump * r) / self.tau_pop
        return drdt, dvdt, dwdt

    def call(self, inputs, states):
        r, v, w, R, U, A = states
        ext = inputs
        h = self.dt_dim

        k1r, k1v, k1w = self._deriv(r, v, w, A)
        k2r, k2v, k2w = self._deriv(r + 0.5 * h * k1r, v + 0.5 * h * k1v, w + 0.5 * h * k1w, A)
        k3r, k3v, k3w = self._deriv(r + 0.5 * h * k2r, v + 0.5 * h * k2v, w + 0.5 * h * k2w, A)
        k4r, k4v, k4w = self._deriv(r + h * k3r, v + h * k3v, w + h * k3w, A)
        r_new = r + (h / 6) * (k1r + 2 * k2r + 2 * k3r + k4r)
        v_new = v + (h / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
        w_new = w + (h / 6) * (k1w + 2 * k2w + 2 * k3w + k4w)

        dt_per_tau = self.dt_dim / self.tau_pop
        FRpre_unit = r * dt_per_tau
        FRpre_ext = ext * 0.001 * self.dt_dim
        FRpre = tf.concat([FRpre_unit, FRpre_ext], axis=1)
        FRpre_full = self.pconn[tf.newaxis, :, :] * FRpre[:, :, tf.newaxis]

        exp_d = tf.exp(-self.dt_dim / self.tau_d)
        exp_r = tf.exp(-self.dt_dim / self.tau_r)
        exp_f = tf.exp(-self.dt_dim / self.tau_f)
        tau1r = tf.where(self.tau_d != self.tau_r,
                         self.tau_d / (self.tau_d - self.tau_r),
                         1e-13)
        exp_d_b = exp_d
        exp_r_b = exp_r
        exp_f_b = exp_f
        tau1r_b = tau1r
        Uinc_b = self.Uinc

        a_ = A * exp_d_b
        r_ = 1.0 + (R - 1.0 + tau1r_b * A) * exp_r_b - tau1r_b * A
        u_ = U * exp_f_b
        released = U * r_ * FRpre_full
        U_new = u_ + Uinc_b * (1.0 - u_) * FRpre_full
        A_new = a_ + released
        R_new = r_ - released

        # Clip state to a finite, biologically plausible range. The Axo unit
        # has tau_pop = 0.36 ms (dt/tau = 0.28, borderline stable for RK4);
        # trained I_ext/gsyn_max can push it past that edge. Clipping here
        # prevents NaN from contaminating the whole simulation while
        # letting training continue on the Border units (indices 0-3).
        r_new = tf.clip_by_value(r_new, 0.0, 200.0)
        v_new = tf.clip_by_value(v_new, -10.0, 10.0)
        w_new = tf.clip_by_value(w_new, -50.0, 50.0)
        R_new = tf.clip_by_value(R_new, 0.0, 1.0)
        U_new = tf.clip_by_value(U_new, 0.0, 1.0)
        A_new = tf.clip_by_value(A_new, 0.0, 1.0)

        output = r_new / (self.tau_pop * 1e-3)
        return output, [r_new, v_new, w_new, R_new, U_new, A_new]


def gather_params():
    pop = build_pop_params()
    gsyn = np.vstack([build_rec_gsyn_matrix(), build_inp_gsyn_matrix()])
    tau_d = np.vstack([build_rec_tau_d_matrix(), build_inp_tau_d_matrix()])
    tau_r = np.vstack([build_rec_tau_r_matrix(), build_inp_tau_r_matrix()])
    tau_f = np.vstack([build_rec_tau_f_matrix(), build_inp_tau_f_matrix()])
    Uinc = np.vstack([build_rec_Uinc_matrix(), build_inp_Uinc_matrix()])
    pconn = np.vstack([build_rec_pconn_matrix(), build_inp_pconn_matrix()])
    e_r = np.vstack([build_rec_e_r_matrix(), build_inp_e_r_matrix()])
    i_ext = np.asarray(pop['I_ext'], dtype=np.float32).copy()
    i_ext[:4] += np.random.uniform(-0.1, 0.1, size=4).astype(np.float32)
    return {
        'alpha': np.asarray(pop['alpha'], dtype=np.float32),
        'a': np.asarray(pop['a'], dtype=np.float32),
        'b': np.asarray(pop['b'], dtype=np.float32),
        'w_jump': np.asarray(pop['w_jump'], dtype=np.float32),
        'tau_pop': np.asarray(pop['tau_pop'], dtype=np.float32),
        'I_ext': i_ext,
        'Delta_I': np.asarray(pop['Delta_I'], dtype=np.float32),
        'gsyn_max': gsyn.astype(np.float32),
        'tau_d': tau_d.astype(np.float32),
        'tau_r': tau_r.astype(np.float32),
        'tau_f': tau_f.astype(np.float32),
        'Uinc': Uinc.astype(np.float32),
        'pconn': pconn.astype(np.float32),
        'e_r': e_r.astype(np.float32),
    }


def decorrelation_penalty(y_pred):
    """Penalize simultaneous activity of all 4 border cells.

    Computes the ratio (off-diagonal sum) / (mean of diagonal) of the
    uncentered 4-cell covariance over time. Scale-invariant in the
    prediction magnitude:
        - one cell active        : 0
        - two cells (corner)     : 4
        - all four active        : 12
    """
    border = y_pred[..., :4]
    T = tf.cast(tf.shape(border)[-2], tf.float32)
    cov = tf.einsum('bti,btj->bij', border, border) / T
    diag = tf.linalg.diag_part(cov)
    off_sum = tf.reduce_sum(cov, axis=[-1, -2]) - tf.reduce_sum(diag, axis=-1)
    denom = tf.reduce_mean(diag, axis=-1) + 1e-6
    return tf.reduce_mean(off_sum / denom)


def build_model(lr = 1e-3, batch_size = 1):
    params = gather_params()
    inputs = Input(shape=(None, config.N_INPUTS), batch_size=batch_size)
    cell = BorderMeanFieldNetwork(params, dt_dim=config.DT, batch_size=batch_size)
    rnn = RNN(cell, return_sequences=True, stateful=True, name='border_rnn')
    out = rnn(inputs)
    model = Model(inputs, out)

    def loss_with_reg(y_true, y_pred):
        return (tf.keras.losses.MSE(    # MeanSquaredLogarithmicError()  cosine_similarity
                    y_true, y_pred[..., :4])
                + config.WTA_WEIGHT * decorrelation_penalty(y_pred))

    model.compile(
        optimizer=Adam(learning_rate=lr, clipvalue=10.0),
        loss=loss_with_reg,
    )
    return model


def load_pretrained(model, path):
    """Assign saved values to matching trainable variables (best-effort)."""
    if not os.path.exists(path):
        print(f"  No pretrained file at {path}, starting from scratch.")
        return 0

    model.load_weights(path)
    # with h5py.File(path, 'r') as f:
    #     if 'parameters' not in f:
    #         print(f"  No 'parameters' group in {path}, starting from scratch.")
    #         return 0
    #     grp = f['parameters']
    #     var_map = {}
    #     for v in model.trainable_variables:
    #         for cand in (v.name,
    #                      v.name.replace(':', '_').replace('/', '_'),
    #                      v.name.split('/')[-1].split(':')[0]):
    #             var_map.setdefault(cand, v)
    #     loaded = 0
    #     for saved_name in grp.keys():
    #         if saved_name not in var_map:
    #             print(f"  WARNING: '{saved_name}' in file but no matching variable in model")
    #             continue
    #         v = var_map[saved_name]
    #         saved_shape = grp[saved_name].shape
    #         if tuple(saved_shape) != tuple(v.shape):
    #             print(f"  WARNING: '{saved_name}' shape mismatch: "
    #                   f"saved={saved_shape}, model={tuple(v.shape)}")
    #             continue
    #         v.assign(grp[saved_name][:])
    #         loaded += 1
    #         print(f"  Loaded {saved_name}: shape={tuple(v.shape)}")
    #print(f"  Loaded {loaded} variable(s) from {path}.")
    #return loaded


def setup_gpu():
    """Configure GPU: log devices, enable memory growth, soft placement."""
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    print(f"  Devices: {len(gpus)} GPU(s), {len(cpus)} CPU(s)")
    for i, gpu in enumerate(gpus):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  GPU:{i} {gpu.name} — memory growth enabled")
        except RuntimeError as e:
            print(f"  GPU:{i} {gpu.name} — memory growth not set: {e}")
    if gpus:
        try:
            tf.config.set_soft_device_placement(True)
        except Exception:
            pass
        try:
            with tf.device('/GPU:0'):
                test = tf.constant([1.0, 2.0, 3.0]) + tf.constant([4.0, 5.0, 6.0])
                _ = test.numpy()
            print(f"  GPU reachable: {test.device}")
        except Exception as e:
            print(f"  WARNING: GPU op failed, falling back to CPU: {e}")
    else:
        print("  WARNING: no GPU visible. Training on CPU.")


def load_all_batches(dataset_path):
    ds = load_dataset_hdf5(dataset_path)
    n_batches = ds['n_batches']
    print(f"  Loading {n_batches} batches into RAM...")
    X_list = []
    Y_list = []
    for i in range(n_batches):
        batch = ds['get_batch'](i)
        X_list.append(batch['inputs'])
        Y_list.append(batch['targets'])
    ds['file'].close()
    X = np.concat(X_list).astype(np.float32)
    Y = np.concat(Y_list).astype(np.float32)
    print(f"  X shape: {X.shape}, Y shape: {Y.shape}, "
          f"X memory: {X.nbytes / 1e6:.1f} MB")
    return X, Y


def save_training_results(loss_history, model):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    h5_path = os.path.join(config.RESULTS_DIR, 'training.h5')
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('loss_history', data=np.array(loss_history))
        grp = f.create_group('parameters')
        used_names = set()
        for idx, v in enumerate(model.trainable_variables):
            base = v.name.replace(':', '_').replace('/', '_') or f'param_{idx}'
            name = base
            suffix = 0
            while name in used_names:
                suffix += 1
                name = f'{base}_{suffix}'
            used_names.add(name)
            grp.create_dataset(name, data=v.numpy())
        cfg = f.create_group('config')
        for attr in dir(config):
            if attr.isupper() and not attr.startswith('_'):
                val = getattr(config, attr)
                if isinstance(val, (int, float, str, bool)):
                    cfg.attrs[attr] = val
    print(f"  HDF5 results saved to {h5_path}")
    json_path = os.path.join(config.RESULTS_DIR, 'training.json')
    payload = {
        'loss_history': [float(x) for x in loss_history],
        'epochs': len(loss_history),
        'grad_method': 'bptt',
        'n_batches_per_epoch': config.N_BATCHES_PER_EPOCH,
        'trainable_variables': [
            {'name': v.name, 'value': v.numpy().tolist()}
            for v in model.trainable_variables
        ],
    }
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON results saved to {json_path}")


def train(dataset_path=None, n_epochs=None, learning_rate=None,
          batches_per_epoch=None, seed=None, resume=None,
          start_batch=None, reset_state_per_epoch=True,
          checkpoint_dir=None):
    """Train the stateful border cell RNN.

    Each epoch processes `batches_per_epoch` consecutive stored batches
    sequentially; the cell's internal state (r, v, w, R, U, A) propagates
    across these calls so the simulation is continuous. State is reset to
    the cell's initial state at the start of each epoch by default
    (set ``reset_state_per_epoch=False`` to keep state across epochs).

    Args:
        dataset_path: path to ``data/dataset.h5``.
        n_epochs: number of epochs (default config.N_EPOCHS).
        learning_rate: Adam LR (default config.LEARNING_RATE).
        batches_per_epoch: sequential batches per epoch (default
            config.N_BATCHES_PER_EPOCH).
        seed: RNG seed (default config.RANDOM_SEED).
        resume: path to a previous weights file to load before training.
        start_batch: index of the first stored batch to feed each epoch.
            ``None`` (default) randomises the start each epoch so the
            network sees different 5 s windows; pass an int to fix it.
        reset_state_per_epoch: if True (default), call
            ``rnn_layer.reset_states()`` at the start of each epoch so
            the network is initialised freshly each time. Set False to
            carry state across epochs.
        checkpoint_dir: where to save per-epoch checkpoints (default
            ``results/checkpoints``).
    """
    ds_path = dataset_path or os.path.join(
        os.path.dirname(config.TRAJECTORY_HDF5), 'dataset.h5')
    n_epochs = n_epochs or config.N_EPOCHS
    lr = learning_rate or config.LEARNING_RATE
    batches_per_epoch = batches_per_epoch or config.N_BATCHES_PER_EPOCH
    if seed is not None:
        config.RANDOM_SEED = seed
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("Configuring devices...")
    setup_gpu()

    print(f"Loading dataset from {ds_path}...")
    X, Y = load_all_batches(ds_path)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    n_batches = X.shape[0]
    if batches_per_epoch > n_batches:
        raise ValueError(
            f"batches_per_epoch ({batches_per_epoch}) > n_batches ({n_batches})")

    print("Building model (stateful RNN, batch_size=1)...")
    model = build_model(lr, batch_size=1)
    n_vars = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"  Trainable parameters: {n_vars}")
    print(f"  Trainable variables: {[v.name for v in model.trainable_variables]}")

    if resume:
        print(f"Resuming from {resume}...")
        load_pretrained(model, resume)

    rnn_layer = model.get_layer('border_rnn')
    checkpoint_dir = checkpoint_dir or os.path.join(
        config.RESULTS_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    max_start = max(0, n_batches - batches_per_epoch)
    log_every = max(1, n_epochs // 20)

    print(f"Training (stateful): {n_epochs} epochs × {batches_per_epoch} "
          f"sequential batches/epoch (out of {n_batches}); "
          f"reset_state_per_epoch={reset_state_per_epoch}...")

    loss_history = []
    t_start = time.time()
    for epoch in range(n_epochs):
        if reset_state_per_epoch:
            rnn_layer.reset_states()

        if start_batch is not None:
            start = start_batch
        elif max_start > 0:
            start = int(np.random.randint(0, max_start + 1))
        else:
            start = 0

        epoch_loss = 0.0
        n_finite = 0
        for i in range(batches_per_epoch):
            x_batch = X[start + i:start + i + 1]
            y_batch = Y[start + i:start + i + 1]
            loss = model.train_on_batch(x_batch, y_batch)
            loss_val = float(loss)
            if not np.isfinite(loss_val):
                print(f"\n  NaN/Inf at epoch {epoch+1}, batch {i+1} "
                      f"(loss={loss_val}), stopping.")
                return loss_history
            epoch_loss += loss_val
            n_finite += 1

        avg_loss = epoch_loss / max(n_finite, 1)
        loss_history.append(avg_loss)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1:4d}/{n_epochs} | loss={avg_loss:.6f} "
                  f"| elapsed={elapsed/60:.1f} min "
                  f"| start_batch={start}")

        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            tag = f"epoch_{epoch + 1:04d}_loss_{avg_loss:.6f}"
            weights_path = os.path.join(checkpoint_dir, f"{tag}.weights.h5")
            model.save_weights(weights_path)
            meta = {
                'epoch': epoch + 1,
                'loss': avg_loss,
                'learning_rate': float(model.optimizer.learning_rate.numpy()),
                'reset_state_per_epoch': reset_state_per_epoch,
            }
            with open(os.path.join(checkpoint_dir, f"{tag}_meta.json"), 'w') as f:
                json.dump(meta, f, indent=2)

    latest_path = os.path.join(checkpoint_dir, 'latest.weights.h5')
    model.save_weights(latest_path)
    history_path = os.path.join(checkpoint_dir, 'loss_history.json')
    with open(history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)

    total_dt = time.time() - t_start
    print(f"Training done in {total_dt/60:.1f} min.")

    return loss_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batches-per-epoch', type=int, default=None,
                        help='Sequential batches per epoch (default '
                             'config.N_BATCHES_PER_EPOCH). The cell state '
                             'propagates across these batches.')
    parser.add_argument('--start-batch', type=int, default=None,
                        help='Fix the starting stored batch for every epoch '
                             '(default: random each epoch).')
    parser.add_argument('--no-reset-state', action='store_true', default=False,
                        help='Carry RNN state across epochs (continuous run) '
                             'instead of resetting at the start of each epoch.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a previous results/checkpoints/latest.weights.h5 to '
                             'load trained weights from before training continues.')
    args = parser.parse_args()
    train(args.dataset, args.epochs, args.lr, args.batches_per_epoch,
          args.seed, args.resume,
          start_batch=args.start_batch,
          reset_state_per_epoch=not args.no_reset_state)


if __name__ == '__main__':
    main()
