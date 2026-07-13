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


class NaNStopping(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        loss = logs.get('loss')
        if loss is None or not np.isfinite(loss):
            print(f"\n  NaN at batch {batch}, stopping.")
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        if loss is None or not np.isfinite(loss):
            print(f"\n  NaN at epoch {epoch}, stopping.")
            self.model.stop_training = True


class R2Metric(tf.keras.metrics.Metric):
    """R² (coefficient of determination) computed per-batch for border cells.

    The model output ``y_pred`` has layout ``[batch, T, units]`` where the first
    four channels are Border_N/S/E/W. We slice those and compare against the
    border-only target ``y_true[..., :4]``, skipping the first
    ``config.LOSS_WARMUP_STEPS`` steps so the synaptic transient doesn't
    dominate the score.
    """

    def __init__(self, name='r2', **kwargs):
        super().__init__(name=name, **kwargs)
        self.ss_res = self.add_weight(name='ss_res', initializer='zeros')
        self.ss_tot = self.add_weight(name='ss_tot', initializer='zeros')

    def reset_state(self):
        self.ss_res.assign(0.0)
        self.ss_tot.assign(0.0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        warmup = config.LOSS_WARMUP_STEPS
        y_true_b = y_true[..., warmup:, :4]
        y_pred_b = y_pred[..., warmup:, :4]
        ss_res = tf.reduce_sum(tf.square(y_true_b - y_pred_b))
        ss_tot = tf.reduce_sum(tf.square(y_true_b - tf.reduce_mean(y_true_b,
                                                                   axis=-2, keepdims=True)))
        self.ss_res.assign_add(ss_res)
        self.ss_tot.assign_add(ss_tot)

    def result(self):
        return 1.0 - self.ss_res / (self.ss_tot + 1e-8)


class R2ValidationCallback(tf.keras.callbacks.Callback):
    """Compute R² on held-out validation data after each epoch."""

    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        warmup = config.LOSS_WARMUP_STEPS
        y_true = tf.constant(self.y_val[..., warmup:, :4], dtype=tf.float32)
        y_pred_full = tf.constant(
            self.model(self.x_val, training=False)[..., warmup:, :],
            dtype=tf.float32)
        y_pred = y_pred_full[..., :4]

        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=-2,
                                                                 keepdims=True)))
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

        per_cell = []
        for c in range(4):
            yt, yp = y_true[..., c], y_pred[..., c]
            s_res = tf.reduce_sum(tf.square(yt - yp))
            s_tot = tf.reduce_sum(tf.square(yt - tf.reduce_mean(yt, axis=-1,
                                                                keepdims=True)))
            per_cell.append(float(1.0 - s_res / (s_tot + 1e-8)))

        names = ['B_N', 'B_S', 'B_E', 'B_W']
        cell_str = '  '.join(f'{n}={r2c:.4f}' for n, r2c in zip(names, per_cell))
        print(f"  [val] R²={float(r2):.4f}  ({cell_str})")


class CheckpointCallback(tf.keras.callbacks.Callback):
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
        self.model.save_weights(
            os.path.join(self.checkpoint_dir, f"{tag}.weights.h5"))
        with open(os.path.join(self.checkpoint_dir, f"{tag}_meta.json"), 'w') as f:
            json.dump({'epoch': epoch + 1, 'loss': loss}, f, indent=2)
        if (epoch + 1) % max(1, self.params.get('epochs', 100) // 20) == 0 or epoch == 0:
            print(f"  [ckpt] epoch {epoch+1}: loss={loss:.6f}")

    def on_train_end(self, logs=None):
        self.model.save_weights(
            os.path.join(self.checkpoint_dir, 'latest.weights.h5'))
        with open(os.path.join(self.checkpoint_dir, 'loss_history.json'), 'w') as f:
            json.dump(self.loss_history, f, indent=2)


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

    def __init__(self, params, dt_dim=0.1, batch_size=1,
                 learnable_init_state=False, **kwargs):
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

        # ── Learnable initial state (r, v, w, R, U, A) ────────────────
        # Default off → get_initial_state() returns random samples (legacy).
        # When on, the initial state is a trainable variable initialised
        # from the BORDER_INIT_* ranges in config.py (defaults reproduce
        # the legacy sampling distributions).
        self._learnable_init_state = bool(learnable_init_state)
        if self._learnable_init_state:
            seed = config.RANDOM_SEED if isinstance(config.RANDOM_SEED, int) else None
            self._r_init = self.add_weight(
                shape=(self.units,),
                initializer=tf.keras.initializers.RandomUniform(
                    minval=config.BORDER_INIT_R_LO,
                    maxval=config.BORDER_INIT_R_HI,
                    seed=seed),
                trainable=True,
                constraint=tf.keras.constraints.NonNeg(),
                name='r_init',
            )
            self._v_init = self.add_weight(
                shape=(self.units,),
                initializer=tf.keras.initializers.RandomNormal(
                    mean=config.BORDER_INIT_V_MEAN,
                    stddev=config.BORDER_INIT_V_STD,
                    seed=seed),
                trainable=True,
                name='v_init',
            )
            self._w_init = self.add_weight(
                shape=(self.units,),
                initializer=tf.constant_initializer(config.BORDER_INIT_W_VAL),
                trainable=True,
                name='w_init',
            )
            self._R_init = self.add_weight(
                shape=(self.pre, self.post),
                initializer=tf.constant_initializer(config.BORDER_INIT_TM_R),
                trainable=True,
                constraint=MinMax(0.0, 1.0),
                name='R_init',
            )
            self._U_init = self.add_weight(
                shape=(self.pre, self.post),
                initializer=tf.constant_initializer(config.BORDER_INIT_TM_U),
                trainable=True,
                constraint=MinMax(0.0, 1.0),
                name='U_init',
            )
            self._A_init = self.add_weight(
                shape=(self.pre, self.post),
                initializer=tf.constant_initializer(config.BORDER_INIT_TM_A),
                trainable=True,
                constraint=MinMax(0.0, 1.0),
                name='A_init',
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
        if self._learnable_init_state:
            return [
                tf.broadcast_to(self._r_init[tf.newaxis, :],
                                [batch_size, self.units]),
                tf.broadcast_to(self._v_init[tf.newaxis, :],
                                [batch_size, self.units]),
                tf.broadcast_to(self._w_init[tf.newaxis, :],
                                [batch_size, self.units]),
                tf.broadcast_to(self._R_init[tf.newaxis, :, :],
                                [batch_size, self.pre, self.post]),
                tf.broadcast_to(self._U_init[tf.newaxis, :, :],
                                [batch_size, self.pre, self.post]),
                tf.broadcast_to(self._A_init[tf.newaxis, :, :],
                                [batch_size, self.pre, self.post]),
            ]

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


def build_model(lr = 1e-3, batch_size = 1, learnable_init_state=False):
    params = gather_params()
    inputs = Input(shape=(None, config.N_INPUTS), batch_size=batch_size)
    cell = BorderMeanFieldNetwork(params, dt_dim=config.DT, batch_size=batch_size,
                                  learnable_init_state=learnable_init_state)
    rnn = RNN(cell, return_sequences=True, stateful=False, name='border_rnn')
    out = rnn(inputs)
    model = Model(inputs, out)

    def loss_with_reg(y_true, y_pred):
        return (tf.keras.losses.MSLE(    # MeanSquaredLogarithmicError()  cosine_similarity
                    y_true, y_pred[..., :4])
                + config.WTA_WEIGHT * decorrelation_penalty(y_pred))

    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        loss=loss_with_reg,
        metrics=[R2Metric()],
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
    """Load the whole dataset into RAM as flat (n_trials, n_steps, ...) arrays.

    Mirrors ``train_wc_nonpsyns.load_all_batches`` — reads ``inputs`` and
    ``targets`` straight from the HDF5 root and returns them as
    ``(n_trials, n_steps, N_INPUTS)`` and ``(n_trials, n_steps, 4)``.
    """
    ds = load_dataset_hdf5(dataset_path)
    X = ds['X']
    Y = ds['Y']
    print(f"  X: {X.shape}, Y: {Y.shape}, {X.nbytes / 1e6:.1f} MB")
    return X, Y


def train(dataset_path=None, n_epochs=None, learning_rate=None,
          seed=None, resume=None, batch_size=None,
          val_split=0.1, learnable_init_state=False):
    """Train the border cell RNN with the built-in Keras ``model.fit`` loop.

    Mirrors ``train_wc_nonpsyns.train``: the last ``val_split`` fraction of
    trials is held out for validation, R² is reported by ``R2Metric`` during
    training and by ``R2ValidationCallback`` after each epoch, and per-epoch
    checkpoints are written by ``CheckpointCallback``. NaN/Inf loss stops
    training immediately via ``NaNStopping``.

    Args:
        dataset_path: path to ``data/dataset.h5`` (default next to the
            trajectory HDF5).
        n_epochs: number of epochs (default ``config.N_EPOCHS``).
        learning_rate: Adam LR (default ``config.LEARNING_RATE``).
        seed: RNG seed (default ``config.RANDOM_SEED``).
        resume: path to a previous ``*.weights.h5`` to load before training.
        batch_size: trials per gradient step (default ``config.BATCH_SIZE``).
        val_split: fraction of trials held out for validation (default 0.1).
        learnable_init_state: pass through to ``BorderMeanFieldNetwork`` —
            makes the initial state trainable instead of randomly sampled.
    """
    ds_path = dataset_path or os.path.join(
        os.path.dirname(config.TRAJECTORY_HDF5), 'dataset.h5')
    n_epochs = n_epochs or config.N_EPOCHS
    lr = learning_rate or config.LEARNING_RATE
    batch_size = batch_size or config.BATCH_SIZE

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

    if X.shape[0] > 10:
        n_val = max(1, int(len(X) * val_split))
        X_val, Y_val = X[-n_val:], Y[-n_val:]
        X_train, Y_train = X[:-n_val], Y[:-n_val]
    else:
        X_val, Y_val = X, Y
        X_train, Y_train = X, Y

    print(f"  Train: {X_train.shape[0]} trials, Val: {X_val.shape[0]} trials")

    print("Building model (RNN, batch_size=%d)..." % batch_size)
    model = build_model(lr=lr, batch_size=batch_size,
                        learnable_init_state=learnable_init_state)
    n_vars = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"  Trainable parameters: {n_vars}")
    for v in model.trainable_variables:
        print(f"    {v.name}: {tuple(v.shape)}")

    if resume and os.path.exists(resume):
        print(f"Resuming from {resume}...")
        model.load_weights(resume)

    callbacks = [
        NaNStopping(),
        CheckpointCallback(),
        # R2ValidationCallback(X_val, Y_val),
    ]

    t_start = time.time()
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks,
    )
    total_dt = time.time() - t_start
    print(f"Training done in {total_dt/60:.1f} min.")

    y_true = tf.constant(Y_val[..., :4], dtype=tf.float32)
    y_pred = tf.constant(model(X_val, training=False)[..., :4])

    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true, axis=-2,
                                                             keepdims=True)))
    final_r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
    print(f"Final validation R² = {final_r2:.4f}")

    return history.history['loss']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a previous results/checkpoints/latest.weights.h5 '
                             'to load trained weights from before training continues.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Trials per gradient step (default config.BATCH_SIZE).')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of trials held out for validation (default: 0.1).')
    parser.add_argument('--learnable-init-state', action='store_true', default=False,
                        help='Make (r, v, w, R, U, A) initial-state components trainable '
                             'variables (default: random sampling each reset).')
    args = parser.parse_args()
    train(args.dataset, args.epochs, args.lr,
          seed=args.seed, resume=args.resume,
          batch_size=args.batch_size, val_split=args.val_split,
          learnable_init_state=args.learnable_init_state)


if __name__ == '__main__':
    main()
