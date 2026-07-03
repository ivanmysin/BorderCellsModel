"""Train border cell network with Wilson-Cowan population model.

Replaces Izhikevich mean-field with Wilson-Cowan rate dynamics:
    τ_i * dE_i/dt = -E_i + S(I_total_i)
    S(x) = M * max(x,0)² / (σ² + max(x,0)²)

TM synapses kept identical to train_simple.py.

Usage:
    python train_wc.py [--dataset data/dataset.h5] [--epochs 100] [--lr 1e-3]
"""
import os

from tensorflow import clip_by_value

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import RNN, Layer
from tensorflow.keras.optimizers import Adam, AdamW
import h5py
import pandas as pd

import config
from utils.params import (
    build_inp_gsyn_matrix, build_inp_tau_f_matrix, build_inp_tau_d_matrix,
    build_inp_tau_r_matrix, build_inp_Uinc_matrix, build_inp_pconn_matrix,
    build_inp_e_r_matrix,
    build_rec_gsyn_matrix, build_rec_tau_f_matrix, build_rec_tau_d_matrix,
    build_rec_tau_r_matrix, build_rec_Uinc_matrix, build_rec_pconn_matrix,
    build_rec_e_r_matrix,
    _get_syn_params, _get_e_r, _UNIT_TYPES,
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
    """R² (coefficient of determination) computed per-batch for border cells."""

    def __init__(self, name='r2', **kwargs):
        super().__init__(name=name, **kwargs)
        self.ss_res = self.add_weight(name='ss_res', initializer='zeros')
        self.ss_tot = self.add_weight(name='ss_tot', initializer='zeros')

    def reset_state(self):
        self.ss_res.assign(0.0)
        self.ss_tot.assign(0.0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        warmup = config.LOSS_WARMUP_STEPS
        # y_pred layout: [E | I_syn]; use the E part for R².
        E_pred = y_pred[..., warmup:, :config.N_POP_UNITS]
        y_true_b = y_true[..., warmup:, :4]
        y_pred_b = E_pred[..., :4]
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
        # Output layout: [E | I_syn]; use the E part for R².
        y_pred = y_pred_full[..., :config.N_POP_UNITS, :4]

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
        with open(os.path.join(self.checkpoint_dir, f"{tag}_meta.json"), 'w') as f:
            json.dump({'epoch': epoch + 1, 'loss': loss}, f, indent=2)
        if (epoch + 1) % max(1, self.params.get('epochs', 100) // 20) == 0 or epoch == 0:
            print(f"  [ckpt] epoch {epoch+1}: loss={loss:.6f}")

    def on_train_end(self, logs=None):
        self.model.save_weights(
            os.path.join(self.checkpoint_dir, 'latest.weights.h5'))
        with open(os.path.join(self.checkpoint_dir, 'loss_history.json'), 'w') as f:
            json.dump(self.loss_history, f, indent=2)


def _softplus(x):
    return tf.nn.softplus(x)


def _inv_softplus(y):
    """Inverse softplus: returns x such that softplus(x) = y.  y > 0."""
    return tf.math.log(tf.math.expm1(tf.maximum(y, 1e-7)))


def _inv_softplus_np(y):
    """NumPy version for weight initialisation."""
    y = np.maximum(y, 1e-7)
    return np.log(np.expm1(y))


def _inv_sigmoid_np(y, lo=0.0, hi=1.0):
    """NumPy: find θ such that sigmoid(θ)*(hi-lo)+lo = y."""
    y_clipped = np.clip(y, lo + 1e-7, hi - 1e-7)
    frac = (y_clipped - lo) / (hi - lo)
    return np.log(frac / (1.0 - frac))


class WilsonCowanNetwork(Layer):
    """Wilson-Cowan rate model + Tsodyks-Markram synapses.

    State: E [batch, units] + R, U, A [batch, pre, post]
    Dynamics: τ_i * dE_i/dt = -E_i + S(Σ gsyn*A*FRpre + I_ext)
    S(x) = M * max(x,0)² / (σ² + max(x,0)²)
    """

    WC_M = 100.0
    WC_SIGMA = 5.0

    # Bounds for Uinc sigmoid mapping
    UINC_LO = 0.04
    UINC_HI = 0.7

    def __init__(self, params, dt_dim=0.1, batch_size=1, n_pre=None, **kwargs):
        super().__init__(**kwargs)
        self.dt_dim = float(dt_dim)
        self.units = config.N_POP_UNITS
        self.pre = n_pre if n_pre is not None else (config.N_POP_UNITS + config.N_INPUTS)
        self.post = config.N_POP_UNITS

        self.pconn = tf.constant(params['pconn'], dtype=tf.float32)
        # self.e_r = tf.constant(params['e_r'], dtype=tf.float32)

        # ei_sign = np.ones((self.pre, self.post), dtype=np.float32)
        # ei_sign[4:6, :] = -1.0
        ei_sign = np.sign(params['e_r'])
        self.ei_sign = tf.constant(ei_sign, dtype=tf.float32)


        wc_tau = np.array([12.0, 12.0, 12.0, 12.0, 10.0, 10.0], dtype=np.float32)
        wc_i_ext = np.ones( self.units, dtype=np.float32 ) + 5.0

        # ── Reparameterised weights ──────────────────────────────────
        # tau_pop: τ = exp(θ), no constraint needed on θ
        theta_tau_pop = np.log(wc_tau)
        self._theta_tau_pop = self.add_weight(
            shape=(self.units,),
            initializer=tf.constant_initializer(theta_tau_pop),
            trainable=False,
            name='theta_tau_pop',
        )

        self.I_ext = self.add_weight(
            shape=(self.units,),
            initializer=tf.constant_initializer(wc_i_ext),
            trainable=config.TRAIN_POP_IEXT,
            name='I_ext',
        )

        # gsyn_max: g = softplus(θ), always > 0
        theta_gsyn = _inv_softplus_np(params['gsyn_max'] * config.SYN_GSYN_INIT_SCALE)
        self._theta_gsyn = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(theta_gsyn),
            trainable=config.TRAIN_SYNAPSE_GMAX,
            name='theta_gsyn',
        )

        # tau_f: τ_f = exp(θ_f)
        theta_tau_f = np.log(np.maximum(params['tau_f'], 1e-7))
        self._theta_tau_f = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(theta_tau_f),
            trainable=config.TRAIN_SYNAPSE_TAU_f,
            name='theta_tau_f',
        )

        # tau_d: τ_d = exp(θ_d)
        theta_tau_d = np.log(np.maximum(params['tau_d'], 1e-7))
        self._theta_tau_d = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(theta_tau_d),
            trainable=config.TRAIN_SYNAPSE_TAU_d,
            name='theta_tau_d',
        )

        # tau_r: τ_r = exp(θ_r)
        theta_tau_r = np.log(np.maximum(params['tau_r'], 1e-7))
        self._theta_tau_r = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(theta_tau_r),
            trainable=config.TRAIN_SYNAPSE_TAU_r,
            name='theta_tau_r',
        )

        # Uinc: U = sigmoid(θ) * (HI - LO) + LO  ∈ (LO, HI)
        theta_Uinc = _inv_sigmoid_np(
            params['Uinc'], self.UINC_LO, self.UINC_HI)
        self._theta_Uinc = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(theta_Uinc),
            trainable=config.TRAIN_SYNAPSE_U,
            name='theta_Uinc',
        )

        self.state_size = [
            tf.TensorShape([batch_size, self.units]),
            tf.TensorShape([batch_size, self.pre, self.post]),
            tf.TensorShape([batch_size, self.pre, self.post]),
            tf.TensorShape([batch_size, self.pre, self.post]),
        ]
        # Output: [E (units) | I_syn (units)] — I_syn is exposed only so that
        # the dead-zone penalty can read it. Dynamics are unchanged.
        self.output_size = self.units * 2

    # ── Transform helpers (θ → physical) ─────────────────────────────
    def _get_tau_pop(self):
        return tf.exp(self._theta_tau_pop)

    def _get_gsyn(self):
        return _softplus(self._theta_gsyn)

    def _get_tau_f(self):
        return tf.exp(self._theta_tau_f)

    def _get_tau_d(self):
        return tf.exp(self._theta_tau_d)

    def _get_tau_r(self):
        return tf.exp(self._theta_tau_r)

    def _get_Uinc(self):
        return tf.sigmoid(self._theta_Uinc) * (self.UINC_HI - self.UINC_LO) + self.UINC_LO

    def get_initial_state(self, batch_size=1):
        return [
            #tf.zeros([batch_size, self.units], dtype=tf.float32),

            tf.random.uniform(
                [batch_size, self.units],
                minval=0.0,
                maxval=3.0,
                dtype=tf.float32,
            ),

            tf.random.uniform(
                [batch_size, self.pre, self.post],
                minval=config.SYN_INIT_R_LO,
                maxval=config.SYN_INIT_R_HI,
                dtype=tf.float32,
            ),
            tf.random.uniform(
                [batch_size, self.pre, self.post],
                minval=config.SYN_INIT_U_LO,
                maxval=config.SYN_INIT_U_HI,
                dtype=tf.float32,
            ),
            tf.random.uniform(
                [batch_size, self.pre, self.post],
                minval=config.SYN_INIT_A_LO,
                maxval=config.SYN_INIT_A_HI,
                dtype=tf.float32,
            ),
        ]

    def _naka_rushton(self, x):
        """S(x) = M * max(x,0)² / (σ² + max(x,0)²)"""
        x_pos = tf.maximum(x, 0.0)
        return self.WC_M * x_pos ** 2 / (self.WC_SIGMA ** 2 + x_pos ** 2)

    def call(self, inputs, states):
        E, R, U, A = states
        ext = inputs
        h = self.dt_dim

        tau_pop = self._get_tau_pop()
        gsyn = self._get_gsyn()
        tau_f = self._get_tau_f()
        tau_d = self._get_tau_d()
        tau_r = self._get_tau_r()
        Uinc = self._get_Uinc()


        FRpre_unit = E * self.dt_dim * 0.001
        FRpre_ext = ext * self.dt_dim * 0.001
        if self.pre == self.units:
            FRpre = FRpre_ext
        else:
            FRpre = tf.concat([FRpre_unit, FRpre_ext], axis=1)
        FRpre_full = self.pconn[tf.newaxis, :, :] * FRpre[:, :, tf.newaxis]

        g_syn = gsyn * A * self.ei_sign[tf.newaxis, :, :]
        I_syn = tf.reduce_sum(g_syn * FRpre_full, axis=1)

        I_total = I_syn + self.I_ext[tf.newaxis, :]
        E_new = E + (h / tau_pop[tf.newaxis, :]) * (
            -E + self._naka_rushton(I_total))

        exp_d = tf.exp(-self.dt_dim / tau_d)
        exp_r = tf.exp(-self.dt_dim / tau_r)
        exp_f = tf.exp(-self.dt_dim / tau_f)
        tau1r = tf.where(tau_d != tau_r,
                         tau_d / (tau_d - tau_r),
                         1e-13)

        a_ = A * exp_d
        r_ = 1.0 + (R - 1.0 + tau1r * A) * exp_r - tau1r * A
        u_ = U * exp_f
        released = U * r_ * FRpre_full
        U_new = tf.clip_by_value(u_ + Uinc * (1.0 - u_) * FRpre_full, 0.0, 1.0)
        A_new = tf.clip_by_value(a_ + released, 0.0, 1.0)
        R_new = tf.clip_by_value(r_ - released, 0.0, 1.0)

        return E_new, [E_new, R_new, U_new, A_new]


def gather_params(n_pre=None):
    if n_pre is None:
        n_pre = config.N_POP_UNITS + config.N_INPUTS
    n_post = config.N_POP_UNITS

    if n_pre == config.N_POP_UNITS + config.N_INPUTS:

        gsyn = np.vstack([build_rec_gsyn_matrix(), build_inp_gsyn_matrix()])
        tau_d = np.vstack([build_rec_tau_d_matrix(), build_inp_tau_d_matrix()])
        tau_r = np.vstack([build_rec_tau_r_matrix(), build_inp_tau_r_matrix()])
        tau_f = np.vstack([build_rec_tau_f_matrix(), build_inp_tau_f_matrix()])
        Uinc = np.vstack([build_rec_Uinc_matrix(), build_inp_Uinc_matrix()])
        pconn = np.vstack([build_rec_pconn_matrix(), build_inp_pconn_matrix()])


        rec_rec = build_rec_e_r_matrix()
        inp_e_r = build_inp_e_r_matrix()
        e_r = np.vstack([rec_rec, inp_e_r])

    elif n_pre == config.N_POP_UNITS + config.N_POP_UNITS:
        # Second layer: 6 recurrent + 6 feedforward = 12→6
        rec_gsyn = build_rec_gsyn_matrix()
        rec_tau_d = build_rec_tau_d_matrix()
        rec_tau_r = build_rec_tau_r_matrix()
        rec_tau_f = build_rec_tau_f_matrix()
        rec_Uinc = build_rec_Uinc_matrix()
        rec_pconn = build_rec_pconn_matrix()
        rec_e_r = build_rec_e_r_matrix()

        p_ff = _get_syn_params('Input→Pyramidal')
        g_base = p_ff['gsyn_max'] * config.GSYN_SCALE_DIMENSIONAL

        ff_gsyn = np.full((n_post, n_post), g_base, dtype=np.float64)
        ff_gsyn *= (1.0 + np.random.uniform(-0.3, 0.3, size=ff_gsyn.shape))
        ff_gsyn = np.maximum(0.001, ff_gsyn)

        ff_tau_d = np.full((n_post, n_post), p_ff['tau_d'], dtype=np.float64)
        ff_tau_r = np.full((n_post, n_post), p_ff['tau_r'], dtype=np.float64)
        ff_tau_f = np.full((n_post, n_post), p_ff['tau_f'], dtype=np.float64)
        ff_Uinc = np.full((n_post, n_post), p_ff['Uinc'], dtype=np.float64)
        ff_pconn = np.ones((n_post, n_post), dtype=np.float64)

        ff_e_r = np.zeros((n_post, n_post), dtype=np.float64)
        for i in range(n_post):
            for j in range(n_post):
                ff_e_r[i, j] = _get_e_r(_UNIT_TYPES[i], _UNIT_TYPES[j])

        gsyn = np.vstack([rec_gsyn, ff_gsyn])
        tau_d = np.vstack([rec_tau_d, ff_tau_d])
        tau_r = np.vstack([rec_tau_r, ff_tau_r])
        tau_f = np.vstack([rec_tau_f, ff_tau_f])
        Uinc = np.vstack([rec_Uinc, ff_Uinc])
        pconn = np.vstack([rec_pconn, ff_pconn])
        e_r = np.vstack([rec_e_r, ff_e_r])

    else:
        gsyn = build_rec_gsyn_matrix().astype(np.float32)
        tau_d = build_rec_tau_d_matrix().astype(np.float32)
        tau_r = build_rec_tau_r_matrix().astype(np.float32)
        tau_f = build_rec_tau_f_matrix().astype(np.float32)
        Uinc = build_rec_Uinc_matrix().astype(np.float32)
        pconn = build_rec_pconn_matrix().astype(np.float32)
        e_r = build_rec_e_r_matrix().astype(np.float32)
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
    """Encourage sparse border-cell activity (winner-take-all).

    At each time step, penalise high mean/max ratio → forces competition
    between border cells so that only one (or a few) fire strongly.
    This creates conditions where inhibitory interneurons are useful.
    """
    border = y_pred[..., :4]                          # [batch, T, 4]
    max_val = tf.reduce_max(border, axis=-1)           # [batch, T]
    mean_val = tf.reduce_mean(border, axis=-1)         # [batch, T]
    sparsity = mean_val / (max_val + 1e-6)            # small ⇒ sparse
    return tf.reduce_mean(sparsity)


def ei_balance_loss(y_pred):
    """Inhibitory activity should scale with excitatory activity.

    Targets: inhibitory_mean = EI_BALANCE_TARGET * excitatory_mean.
    Gives gradient to Basket/Axo even when they start near zero.
    """
    exc = y_pred[..., :4]                              # border cells
    inhib = y_pred[..., 4:6]                           # Basket, Axo
    exc_mean = tf.reduce_mean(exc, axis=[-1, -2])
    inhib_mean = tf.reduce_mean(inhib, axis=[-1, -2])
    target_ratio = 0.6
    actual_ratio = exc_mean / (exc_mean + inhib_mean  + 1e-6)
    return tf.reduce_mean((actual_ratio - target_ratio) ** 2)


def synapse_dead_zone_penalty(I_syn):
    """Push I_syn out of the Naka-Rushton dead zone (I_syn ~ 0).

    Softplus(-(I_syn - threshold) / tau) is large when I_syn <= threshold,
    small otherwise. Gradient is non-zero everywhere (including at I_syn=0),
    so it provides a path I_syn -> gsyn that bypasses S'(I_syn)=0.

    Softplus derivative:  d/dx softplus(x) = sigmoid(x) > 0 for all x.
    """
    z = -(I_syn - config.SYN_DEAD_ZONE_THRESHOLD) / config.SYN_DEAD_ZONE_TAU
    return config.SYN_DEAD_ZONE_WEIGHT * tf.reduce_mean(tf.nn.softplus(z))


def build_model(lr=1e-3, batch_size=1, n_layers=2):
    inputs = Input(shape=(None, config.N_INPUTS), batch_size=batch_size)

    params1 = gather_params(n_pre=config.N_POP_UNITS + config.N_INPUTS)




    cell1 = WilsonCowanNetwork(params1, dt_dim=config.DT, batch_size=batch_size,
                               name='wc_layer1')
    # # 📊 Сохранение параметров в Excel
    # print("💾 Saving Wilson-Cowan parameters to Excel...")
    # with pd.ExcelWriter("wc_layer1_params.xlsx", engine="openpyxl") as writer:
    #     for name, value in params1.items():
    #         # name = v.name.split(':')[0]  # remove ':0' suffix
    #         # value = v.numpy()
    #         # Handle 0D scalars
    #         if value.ndim == 0:
    #             df = pd.DataFrame([[value.item()]], columns=[name])
    #         else:
    #             df = pd.DataFrame(value)
    #         df.to_excel(writer, sheet_name=name[:31], index=False)  # Excel limit: 31 chars
    # print("✅ Saved to wc_layer1_params.xlsx")
    #
    # assert(False)

    x = RNN(cell1, return_sequences=True, stateful=False, name='wc_rnn1')(inputs)

    if n_layers >= 2:
        n_pre2 = config.N_POP_UNITS + config.N_POP_UNITS
        params2 = gather_params(n_pre=n_pre2)

        # # 📊 Сохранение параметров в Excel
        # print("💾 Saving Wilson-Cowan parameters to Excel...")
        # with pd.ExcelWriter("wc_layer2_params.xlsx", engine="openpyxl") as writer:
        #     for name, value in params2.items():
        #         # name = v.name.split(':')[0]  # remove ':0' suffix
        #         # value = v.numpy()
        #         # Handle 0D scalars
        #         if value.ndim == 0:
        #             df = pd.DataFrame([[value.item()]], columns=[name])
        #         else:
        #             df = pd.DataFrame(value)
        #         df.to_excel(writer, sheet_name=name[:31], index=False)  # Excel limit: 31 chars
        # print("✅ Saved to wc_layer2_params.xlsx")
        #
        # assert(False)

        cell2 = WilsonCowanNetwork(params2, dt_dim=config.DT, batch_size=batch_size,
                                   n_pre=n_pre2, name='wc_layer2')
        # Layer 1 output is [E | I_syn]; pass only E to layer 2 so its
        # recurrent/feedforward synapses receive firing rates, not currents.
        E_from_layer1 = x[..., :config.N_POP_UNITS]
        x = RNN(cell2, return_sequences=True, stateful=False, name='wc_rnn2')(E_from_layer1)

    model = Model(inputs, x)

    def loss_with_reg(y_true, y_pred):
        warmup = config.LOSS_WARMUP_STEPS
        y_true_w = y_true[..., warmup:, :]
        y_pred_w = y_pred[..., warmup:, :]

        L_mse = tf.keras.losses.MeanSquaredError()(y_true_w, y_pred_w[..., :4])
        # L_wta = config.WTA_WEIGHT * decorrelation_penalty(E_pred)
        # L_sharp = config.LOSS_WEIGHT_SHARPENING * sharpening_loss(E_pred)
        # L_ei = config.LOSS_WEIGHT_EI_BALANCE * ei_balance_loss(E_pred)

        return L_mse # + L_wta + L_sharp + L_ei

    model.compile(
        optimizer=Adam(learning_rate=lr, clipvalue=15.0),
        loss=loss_with_reg,
        metrics=[R2Metric()],
    )
    return model


def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    print(f"  Devices: {len(gpus)} GPU(s), {len(cpus)} CPU(s)")
    for i, gpu in enumerate(gpus):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  GPU:{i} {gpu.name} — memory growth enabled")
        except RuntimeError:
            pass
    if gpus:
        try:
            tf.config.set_soft_device_placement(True)
        except Exception:
            pass
    else:
        print("  WARNING: no GPU visible. Training on CPU.")


def load_all_batches(dataset_path):
    ds = load_dataset_hdf5(dataset_path)
    n_batches = ds['n_batches']
    print(f"  Loading {n_batches} batches into RAM...")
    X_list, Y_list = [], []
    for i in range(n_batches):
        b = ds['get_batch'](i)
        X_list.append(b['inputs'])
        Y_list.append(b['targets'])
    ds['file'].close()
    X = np.concat(X_list).astype(np.float32)
    Y = np.concat(Y_list).astype(np.float32)
    print(f"  X: {X.shape}, Y: {Y.shape}, {X.nbytes / 1e6:.1f} MB")
    return X, Y


def train(dataset_path=None, n_epochs=None, learning_rate=None,
          batches_per_epoch=None, seed=None, resume=None, n_layers=1, batch_size=None,
          val_split=0.1):
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

    if X.shape[0] > 10:
        n_val = max(1, int(len(X) * val_split))
        X_val, Y_val = X[-n_val:], Y[-n_val:]
        X_train, Y_train = X[:-n_val], Y[:-n_val]
    else:
        X_val, Y_val = X, Y
        X_train, Y_train = X, Y

    print(f"  Train: {X_train.shape[0]} batches, Val: {X_val.shape[0]} batches")

    print(f"Building Wilson-Cowan model ({n_layers} layers)...")
    model = build_model(lr=lr, batch_size=batch_size, n_layers=n_layers)
    n_vars = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"  Trainable parameters: {n_vars}")
    for v in model.trainable_variables:
        print(f"    {v.name}: {tuple(v.shape)}")

    if resume and os.path.exists(resume):
        print(f"Resuming from {resume}...")
        model.load_weights(resume)

    callbacks = [NaNStopping(), CheckpointCallback()] #, R2ValidationCallback(X_val, Y_val)]

    t_start = time.time()
    history = model.fit(
        X_train, Y_train,
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks,
    )
    total_dt = time.time() - t_start
    print(f"Training done in {total_dt/60:.1f} min.")

    warmup = config.LOSS_WARMUP_STEPS
    y_true = tf.constant(Y_val[:, :, :4], dtype=tf.float32)
    y_pred = tf.constant(model(X_val, training=False)[:, :, :4])


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
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--layers', type=int, default=1,
                        help='Number of WC layers (default: 1)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of data for validation (default: 0.1)')
    args = parser.parse_args()
    train(args.dataset, args.epochs, args.lr, seed=args.seed, resume=args.resume,
          n_layers=args.layers, batch_size=args.batch_size, val_split=args.val_split)


if __name__ == '__main__':
    main()
