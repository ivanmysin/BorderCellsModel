"""Train border cell network with Wilson-Cowan population model.

Replaces Izhikevich mean-field with Wilson-Cowan rate dynamics:
    τ_i * dE_i/dt = -E_i + S(I_total_i)
    S(x) = M * max(x,0)² / (σ² + max(x,0)²)

TM synapses kept identical to train_simple.py.

Usage:
    python train_wc.py [--dataset data/dataset.h5] [--epochs 100] [--lr 1e-3]
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
import h5py

import config
from utils.params import (
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
        self.e_r = tf.constant(params['e_r'], dtype=tf.float32)

        ei_sign = np.ones((self.pre, self.post), dtype=np.float32)
        ei_sign[4:6, :] = -1.0
        self.ei_sign = tf.constant(ei_sign, dtype=tf.float32)


        wc_tau = np.array([12.0, 12.0, 12.0, 12.0, 10.0, 10.0], dtype=np.float32)
        wc_i_ext = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

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
        theta_gsyn = _inv_softplus_np(params['gsyn_max'])
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
        self.output_size = self.units

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
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.ones([batch_size, self.pre, self.post], dtype=tf.float32),
            tf.zeros([batch_size, self.pre, self.post], dtype=tf.float32),
            tf.zeros([batch_size, self.pre, self.post], dtype=tf.float32),
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


        FRpre_unit = E * self.dt_dim * 0.001 #  tf.clip_by_value(, 0.0, 0.5)
        FRpre_ext =  ext * self.dt_dim * 0.001 #tf.clip_by_value(ext * 0.1, 0.0, 0.5)
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
        e_r = np.vstack([build_rec_e_r_matrix(), build_inp_e_r_matrix()])
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
    exc_mean = tf.reduce_mean(tf.abs(exc), axis=[-1, -2])
    inhib_mean = tf.reduce_mean(tf.abs(inhib), axis=[-1, -2])
    target_ratio = 0.3
    actual_ratio = inhib_mean / (exc_mean + 1e-6)
    return tf.reduce_mean((actual_ratio - target_ratio) ** 2)


def build_model(lr=1e-3, batch_size=1, n_layers=2):
    inputs = Input(shape=(None, config.N_INPUTS), batch_size=batch_size)

    params1 = gather_params(n_pre=config.N_POP_UNITS + config.N_INPUTS)
    params1['gsyn_max'] *= 5
    cell1 = WilsonCowanNetwork(params1, dt_dim=config.DT, batch_size=batch_size,
                               name='wc_layer1')
    x = RNN(cell1, return_sequences=True, stateful=False, name='wc_rnn1')(inputs)

    if n_layers >= 2:
        params2 = gather_params(n_pre=config.N_POP_UNITS)
        cell2 = WilsonCowanNetwork(params2, dt_dim=config.DT, batch_size=batch_size,
                                   n_pre=config.N_POP_UNITS, name='wc_layer2')
        x = RNN(cell2, return_sequences=True, stateful=False, name='wc_rnn2')(x)

    model = Model(inputs, x)

    def loss_with_reg(y_true, y_pred):
        L_mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred[..., :4])
        L_wta = config.WTA_WEIGHT * decorrelation_penalty(y_pred)
        L_sharp = config.LOSS_WEIGHT_SHARPENING * sharpening_loss(y_pred)
        L_ei = config.LOSS_WEIGHT_EI_BALANCE * ei_balance_loss(y_pred)
        return L_mse + L_wta + L_sharp + L_ei

    model.compile(
        optimizer=Adam(learning_rate=lr, clipnorm=1.0),
        loss=loss_with_reg,
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
          batches_per_epoch=None, seed=None, resume=None, n_layers=1, batch_size=None):
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

    print(f"Building Wilson-Cowan model ({n_layers} layers)...")
    model = build_model(lr=lr, batch_size=batch_size, n_layers=n_layers)
    n_vars = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"  Trainable parameters: {n_vars}")
    for v in model.trainable_variables:
        print(f"    {v.name}: {tuple(v.shape)}")

    if resume and os.path.exists(resume):
        print(f"Resuming from {resume}...")
        model.load_weights(resume)

    callbacks = [NaNStopping(), CheckpointCallback()]

    t_start = time.time()
    history = model.fit(
        X, Y,
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks,
    )
    total_dt = time.time() - t_start
    print(f"Training done in {total_dt/60:.1f} min.")
    return history.history['loss']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--layers', type=int, default=2,
                        help='Number of WC layers (default: 2)')
    args = parser.parse_args()
    train(args.dataset, args.epochs, args.lr, seed=args.seed, resume=args.resume,
          n_layers=args.layers, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
