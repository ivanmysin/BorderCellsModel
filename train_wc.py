"""Train border cell network: Wilson-Cowan population + Tsodyks-Markram synapses.

Population model (from ``train_wc_nonpsyns.py``):
    τ_i * dE_i/dt = -E_i + S(I_total_i)
    S(x) = M * max(x, 0)² / (σ² + max(x, 0)²)

Synapse model (from ``train_simple.py``):
    Tsodyks-Markram 3-state (R, U, A) with depression / facilitation /
    resource depletion. ``I_syn = gsyn * A * ei_sign`` (rate-model form,
    no reversal-potential term). State is clipped to [0, 1] per step.

Inputs are converted to spikes/ms by the 0.001 factor (Hz → kHz) — this
keeps the synapse equation dimensionally consistent with the Izhikevich
model in ``train_simple.py``.

Usage:
    python train_wc.py [--dataset data/dataset.h5] [--epochs 100]
                       [--lr 1e-3] [--seed 42] [--batch_size 5]
                       [--val_split 0.1] [--learnable-init-state]
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import NonNeg

import config
from utils.params import (
    build_rec_gsyn_matrix, build_rec_tau_d_matrix, build_rec_tau_r_matrix,
    build_rec_tau_f_matrix, build_rec_Uinc_matrix, build_rec_pconn_matrix,
    build_rec_e_r_matrix,
    build_inp_gsyn_matrix, build_inp_tau_d_matrix, build_inp_tau_r_matrix,
    build_inp_tau_f_matrix, build_inp_Uinc_matrix, build_inp_pconn_matrix,
    build_inp_e_r_matrix,
)
from utils.dataset import load_dataset_hdf5


class MinMaxCliper(tf.keras.constraints.Constraint):
    """Clip weights element-wise to [min_val, max_val] (broadcastable)."""

    def __init__(self, min_val, max_val):
        self.min_val = tf.constant(min_val, dtype=tf.float32)
        self.max_val = tf.constant(max_val, dtype=tf.float32)

    def __call__(self, w):
        return tf.clip_by_value(
            w,
            tf.broadcast_to(self.min_val, tf.shape(w)),
            tf.broadcast_to(self.max_val, tf.shape(w)),
        )


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
    """R² computed per-batch on border cells (skip warmup)."""

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
        ss_tot = tf.reduce_sum(tf.square(
            y_true_b - tf.reduce_mean(y_true_b, axis=-2, keepdims=True)))
        self.ss_res.assign_add(ss_res)
        self.ss_tot.assign_add(ss_tot)

    def result(self):
        return 1.0 - self.ss_res / (self.ss_tot + 1e-8)


class R2ValidationCallback(tf.keras.callbacks.Callback):
    """R² on held-out validation data after each epoch.

    Uses ``model.predict`` (compiled forward pass) instead of a direct
    ``model(x)`` call in eager mode — eager forward through a 100k-step
    RNN is ~10× slower than the compiled version that ``model.fit`` uses.
    """

    def __init__(self, x_val, y_val, batch_size):
        super().__init__()
        self.x_val = x_val
        self.batch_size = batch_size
        # Pre-compute the warmup-sliced target once; it never changes.
        self._y_true = tf.constant(
            y_val[..., config.LOSS_WARMUP_STEPS:, :4], dtype=tf.float32)

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(
            self.x_val, batch_size=self.batch_size, verbose=0
        )[..., config.LOSS_WARMUP_STEPS:, :4]

        ss_res = tf.reduce_sum(tf.square(self._y_true - y_pred))
        ss_tot = tf.reduce_sum(tf.square(
            self._y_true - tf.reduce_mean(self._y_true, axis=-2, keepdims=True)))
        r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

        # Batched per-cell R²: one reduce over [n_val, T, 4] instead of a
        # Python loop over 4 cells.
        yt_mean = tf.reduce_mean(self._y_true, axis=-2, keepdims=True)
        s_res = tf.reduce_sum(
            tf.square(self._y_true - y_pred), axis=[-3, -2])
        s_tot = tf.reduce_sum(
            tf.square(self._y_true - yt_mean), axis=[-3, -2])
        per_cell = (1.0 - s_res / (s_tot + 1e-8)).numpy().tolist()

        names = ['B_N', 'B_S', 'B_E', 'B_W']
        cell_str = '  '.join(
            f'{n}={r2c:.4f}' for n, r2c in zip(names, per_cell))
        print(f"  [val] R²={r2:.4f}  ({cell_str})")


class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir=None):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            config.RESULTS_DIR, 'checkpoints_wc')
        self.loss_history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = float(logs.get('loss', float('nan')))
        self.loss_history.append(loss)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        tag = f"epoch_{epoch + 1:04d}_loss_{loss:.6f}"
        self.model.save_weights(
            os.path.join(self.checkpoint_dir, f"{tag}.weights.h5"))
        with open(os.path.join(self.checkpoint_dir, f"{tag}_meta.json"),
                  'w') as f:
            json.dump({'epoch': epoch + 1, 'loss': loss}, f, indent=2)
        if ((epoch + 1) % max(1, self.params.get('epochs', 100) // 20) == 0
                or epoch == 0):
            print(f"  [ckpt] epoch {epoch+1}: loss={loss:.6f}")

    def on_train_end(self, logs=None):
        self.model.save_weights(
            os.path.join(self.checkpoint_dir, 'latest.weights.h5'))
        with open(os.path.join(self.checkpoint_dir, 'loss_history.json'),
                  'w') as f:
            json.dump(self.loss_history, f, indent=2)


class WilsonCowanNetwork(Layer):
    """Wilson-Cowan rate population + Tsodyks-Markram plastic synapses.

    State: E [batch, units] + R, U, A [batch, pre, post]
    Population: τ_i * dE_i/dt = -E_i + S(I_syn + I_ext)
    Synapse:   I_syn = gsyn * A * ei_sign  (rate-model form)
               R, U, A evolve per standard TM update; clipped to [0, 1].
    """

    WC_M = 100.0
    WC_SIGMA = 5.0

    def __init__(self, params, dt_dim=0.1, batch_size=1, n_pre=None,
                 learnable_init_state=False, **kwargs):
        super().__init__(**kwargs)
        self.dt_dim = float(dt_dim)
        self.units = config.N_POP_UNITS
        self.pre = n_pre if n_pre is not None else (
            config.N_POP_UNITS + config.N_INPUTS)
        self.post = config.N_POP_UNITS

        self.pconn = tf.constant(params['pconn'], dtype=tf.float32)
        self.ei_sign = tf.constant(np.sign(params['e_r']), dtype=tf.float32)

        # Population params (hardcoded, from train_wc_nonpsyns.py)
        wc_tau = np.array([20.0, 20.0, 20.0, 20.0, 10.0, 10.0],
                          dtype=np.float32)
        wc_i_ext = np.array(
            [config.BORDER_INIT_I_EXT] * 4
            + [config.BASKET_INIT_I_EXT, config.AXO_INIT_I_EXT],
            dtype=np.float32,
        )

        self.tau_pop = self.add_weight(
            shape=(self.units,),
            initializer=tf.constant_initializer(wc_tau),
            trainable=False,
            name='tau_pop',
        )
        self.I_ext = self.add_weight(
            shape=(self.units,),
            initializer=tf.constant_initializer(wc_i_ext),
            trainable=config.TRAIN_POP_IEXT,
            name='I_ext',
        )

        # Synapse params (direct weights + constraints, no reparameterisation)
        self.gsyn_max = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(config.GSYN_SCALE_DIMENSIONAL * params['gsyn_max']),
            trainable=config.TRAIN_SYNAPSE_GMAX,
            constraint=NonNeg(),
            name='gsyn_max',
        )
        self.tau_d = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(params['tau_d']),
            trainable=config.TRAIN_SYNAPSE_TAU_d,
            constraint=MinMaxCliper(min_val=2.0, max_val=15.0),
            name='tau_d',
        )
        self.tau_r = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(params['tau_r']),
            trainable=config.TRAIN_SYNAPSE_TAU_r,
            constraint=MinMaxCliper(min_val=91.0, max_val=1300.0),
            name='tau_r',
        )
        self.tau_f = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(params['tau_f']),
            trainable=config.TRAIN_SYNAPSE_TAU_f,
            constraint=MinMaxCliper(min_val=6.0, max_val=240.0),
            name='tau_f',
        )
        self.Uinc = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(params['Uinc']),
            trainable=config.TRAIN_SYNAPSE_U,
            constraint=MinMaxCliper(min_val=0.04, max_val=0.7),
            name='Uinc',
        )

        # Learnable initial state (default off → zeros)
        self._learnable_init_state = bool(learnable_init_state)
        if self._learnable_init_state:
            seed = (config.RANDOM_SEED
                    if isinstance(config.RANDOM_SEED, int) else None)
            e_init_values = np.zeros(self.units, dtype=np.float32)
            e_init_values[self.units - 1] = config.AXO_INIT_RATE
            self._E_init = self.add_weight(
                shape=(self.units,),
                initializer=tf.constant_initializer(e_init_values),
                trainable=True, constraint=NonNeg(), name='E_init',
            )
            self._R_init = self.add_weight(
                shape=(self.pre, self.post),
                initializer=tf.keras.initializers.RandomUniform(
                    minval=config.SYN_INIT_R_LO,
                    maxval=config.SYN_INIT_R_HI, seed=seed),
                trainable=True,
                constraint=MinMaxCliper(0.0, 1.0), name='R_init',
            )
            self._U_init = self.add_weight(
                shape=(self.pre, self.post),
                initializer=tf.keras.initializers.RandomUniform(
                    minval=config.SYN_INIT_U_LO,
                    maxval=config.SYN_INIT_U_HI, seed=seed),
                trainable=True,
                constraint=MinMaxCliper(0.0, 1.0), name='U_init',
            )
            self._A_init = self.add_weight(
                shape=(self.pre, self.post),
                initializer=tf.keras.initializers.RandomUniform(
                    minval=config.SYN_INIT_A_LO,
                    maxval=config.SYN_INIT_A_HI, seed=seed),
                trainable=True,
                constraint=MinMaxCliper(0.0, 1.0), name='A_init',
            )

        self.state_size = [
            tf.TensorShape([self.units]),
            tf.TensorShape([self.pre, self.post]),
            tf.TensorShape([self.pre, self.post]),
            tf.TensorShape([self.pre, self.post]),
        ]
        self.output_size = self.units

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None:
            batch_size = 1
        if self._learnable_init_state:
            return [
                tf.broadcast_to(self._E_init[tf.newaxis, :],
                                [batch_size, self.units]),
                tf.broadcast_to(self._R_init[tf.newaxis, :, :],
                                [batch_size, self.pre, self.post]),
                tf.broadcast_to(self._U_init[tf.newaxis, :, :],
                                [batch_size, self.pre, self.post]),
                tf.broadcast_to(self._A_init[tf.newaxis, :, :],
                                [batch_size, self.pre, self.post]),
            ]
        # Default (non-learnable) initial state:
        # E=0 for borders and Basket, E=AXO_INIT_RATE for Axo (agent starts
        # at the arena center, so Axo is in its active state from t=0).
        e_init = tf.zeros([batch_size, self.units], dtype=tf.float32)
        e_init = e_init + tf.one_hot(self.units - 1, self.units) * tf.constant(
            config.AXO_INIT_RATE, dtype=tf.float32
        )
        return [
            e_init,
            tf.random.uniform(
                [batch_size, self.pre, self.post],
                minval=config.SYN_INIT_R_LO, maxval=config.SYN_INIT_R_HI,
                dtype=tf.float32,
            ),
            tf.random.uniform(
                [batch_size, self.pre, self.post],
                minval=config.SYN_INIT_U_LO, maxval=config.SYN_INIT_U_HI,
                dtype=tf.float32,
            ),
            tf.random.uniform(
                [batch_size, self.pre, self.post],
                minval=config.SYN_INIT_A_LO, maxval=config.SYN_INIT_A_HI,
                dtype=tf.float32,
            ),
        ]

    def _naka_rushton(self, x):
        x_pos = tf.maximum(x, 0.0)
        return self.WC_M * x_pos ** 2 / (self.WC_SIGMA ** 2 + x_pos ** 2)

    def call(self, inputs, states):
        E, R, U, A = states
        h = self.dt_dim

        # Presynaptic firing rate → spikes/ms (Hz * 0.001 = kHz).
        FRpre_unit = E * 0.001 * h
        FRpre_ext = inputs * 0.001 * h
        if self.pre == self.units:
            FRpre = FRpre_ext
        else:
            FRpre = tf.concat([FRpre_unit, FRpre_ext], axis=1)
        FRpre_full = self.pconn[tf.newaxis, :, :] * FRpre[:, :, tf.newaxis]

        # Synaptic current (rate-model form, no reversal potential).
        g_syn = self.gsyn_max * A * self.ei_sign[tf.newaxis, :, :]
        I_syn = tf.reduce_sum(g_syn, axis=1)

        I_total = I_syn + self.I_ext[tf.newaxis, :]
        E_new = E + (h / self.tau_pop[tf.newaxis, :]) * (
            -E + self._naka_rushton(I_total))

        # Tsodyks-Markram update.
        exp_d = tf.exp(-h / self.tau_d)
        exp_r = tf.exp(-h / self.tau_r)
        exp_f = tf.exp(-h / self.tau_f)
        tau1r = tf.where(self.tau_d != self.tau_r,
                         self.tau_d / (self.tau_d - self.tau_r), 1e-13)

        a_ = A * exp_d
        r_ = 1.0 + (R - 1.0 + tau1r * A) * exp_r - tau1r * A
        u_ = U * exp_f
        released = U * r_ * FRpre_full
        U_new = u_ + self.Uinc * (1.0 - u_) * FRpre_full
        A_new = a_ + released
        R_new = r_ - released

        # Clip TM state to [0, 1].
        R_new = tf.clip_by_value(R_new, 0.0, 1.0)
        U_new = tf.clip_by_value(U_new, 0.0, 1.0)
        A_new = tf.clip_by_value(A_new, 0.0, 1.0)

        return E_new, [E_new, R_new, U_new, A_new]


def gather_params(n_pre=None):
    """Build parameter matrices from CSV (recurrent + input synapses)."""
    if n_pre is None:
        n_pre = config.N_POP_UNITS + config.N_INPUTS

    gsyn = np.vstack([build_rec_gsyn_matrix(), build_inp_gsyn_matrix()])
    tau_d = np.vstack([build_rec_tau_d_matrix(), build_inp_tau_d_matrix()])
    tau_r = np.vstack([build_rec_tau_r_matrix(), build_inp_tau_r_matrix()])
    tau_f = np.vstack([build_rec_tau_f_matrix(), build_inp_tau_f_matrix()])
    Uinc = np.vstack([build_rec_Uinc_matrix(), build_inp_Uinc_matrix()])
    pconn = np.vstack([build_rec_pconn_matrix(), build_inp_pconn_matrix()])
    e_r = np.vstack([build_rec_e_r_matrix(), build_inp_e_r_matrix()])

    return {
        'gsyn_max': gsyn.astype(np.float32),
        'tau_d': tau_d.astype(np.float32),
        'tau_r': tau_r.astype(np.float32),
        'tau_f': tau_f.astype(np.float32),
        'Uinc': Uinc.astype(np.float32),
        'pconn': pconn.astype(np.float32),
        'e_r': e_r.astype(np.float32),
    }


def decorrelation_penalty(y_pred):
    border = y_pred[..., :4]
    T = tf.cast(tf.shape(border)[-2], tf.float32)
    cov = tf.einsum('bti,btj->bij', border, border) / T
    diag = tf.linalg.diag_part(cov)
    off_sum = tf.reduce_sum(cov, axis=[-1, -2]) - tf.reduce_sum(diag, axis=-1)
    denom = tf.reduce_mean(diag, axis=-1) + 1e-6
    return tf.reduce_mean(off_sum / denom)


def sharpening_loss(y_pred):
    """Sparse border-cell activity (winner-take-all)."""
    border = y_pred[..., :4]
    max_val = tf.reduce_max(border, axis=-1)
    mean_val = tf.reduce_mean(border, axis=-1)
    return tf.reduce_mean(mean_val / (max_val + 1e-6))


def ei_balance_loss(y_pred):
    """Inhibitory activity ∝ excitatory activity."""
    exc = y_pred[..., :4]
    inhib = y_pred[..., 4:6]
    exc_mean = tf.reduce_mean(exc, axis=[-1, -2])
    inhib_mean = tf.reduce_mean(inhib, axis=[-1, -2])
    target_ratio = 0.6
    actual_ratio = exc_mean / (exc_mean + inhib_mean + 1e-6)
    return tf.reduce_mean((actual_ratio - target_ratio) ** 2)


def build_model(lr=1e-3, batch_size=1, learnable_init_state=False):
    params = gather_params()
    inputs = Input(shape=(None, config.N_INPUTS), batch_size=batch_size)
    cell = WilsonCowanNetwork(
        params, dt_dim=config.DT, batch_size=batch_size,
        learnable_init_state=learnable_init_state, name='wc_layer',
    )
    x = RNN(cell, return_sequences=True, stateful=False, name='wc_rnn')(inputs)
    model = Model(inputs, x)

    def loss_with_reg(y_true, y_pred):
        L_mse = tf.keras.losses.MSE(y_true, y_pred[..., :4]) + 10 * tf.keras.losses.cosine_similarity(y_true, y_pred[..., :4])
        L_wta = config.WTA_WEIGHT * decorrelation_penalty(y_pred)
        L_sharp = config.LOSS_WEIGHT_SHARPENING * sharpening_loss(y_pred)
        L_ei = config.LOSS_WEIGHT_EI_BALANCE * ei_balance_loss(y_pred)
        return L_mse + L_wta + L_sharp + L_ei

    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        loss=loss_with_reg,
        metrics=[R2Metric()],
    )
    return model


def load_all_batches(dataset_path):
    """Load flat (n_trials, n_steps, ...) arrays from the HDF5 dataset."""
    ds = load_dataset_hdf5(dataset_path)
    X = ds['X']
    Y = ds['Y']
    print(f"  X: {X.shape}, Y: {Y.shape}, {X.nbytes / 1e6:.1f} MB")
    return X, Y


def train(dataset_path=None, n_epochs=None, learning_rate=None,
          seed=None, resume=None, batch_size=None,
          val_split=0.1, learnable_init_state=False):
    ds_path = dataset_path or os.path.join(
        os.path.dirname(config.TRAJECTORY_HDF5), 'dataset.h5')
    n_epochs = n_epochs or config.N_EPOCHS
    lr = learning_rate or config.LEARNING_RATE
    batch_size = batch_size or config.BATCH_SIZE

    if seed is not None:
        config.RANDOM_SEED = seed
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print(f"Loading dataset from {ds_path}...")
    X, Y = load_all_batches(ds_path)

    if X.shape[0] > 10:
        n_val = max(1, int(len(X) * val_split))
        X_val, Y_val = X[-n_val:], Y[-n_val:]
        X_train, Y_train = X[:-n_val], Y[:-n_val]
    else:
        X_val, Y_val = X, Y
        X_train, Y_train = X, Y

    print(f"  Train: {X_train.shape[0]} trials, Val: {X_val.shape[0]} trials")

    print("Building Wilson-Cowan + TM-synapse model...")
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
        # R2ValidationCallback(X_val, Y_val, batch_size=batch_size),
    ]

    t_start = time.time()
    history = model.fit(
        X_train, Y_train,
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks,
        validation_data=(X_val, Y_val),
    )
    total_dt = time.time() - t_start
    print(f"Training done in {total_dt/60:.1f} min.")

    y_pred = model.predict(
        X_val, batch_size=batch_size, verbose=0)[..., :4]
    y_true = Y_val[..., :4]

    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(
        y_true - tf.reduce_mean(y_true, axis=-2, keepdims=True)))
    final_r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
    print(f"Final validation R² = {final_r2:.4f}")

    return history.history['loss']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of trials held out for validation (default: 0.1).')
    parser.add_argument('--learnable-init-state', action='store_true',
                        default=False,
                        help='Make (E, R, U, A) initial-state components trainable '
                             '(default: zeros/random sampling).')
    args = parser.parse_args()
    train(args.dataset, args.epochs, args.lr, seed=args.seed,
          resume=args.resume, batch_size=args.batch_size,
          val_split=args.val_split,
          learnable_init_state=args.learnable_init_state)


if __name__ == '__main__':
    main()
