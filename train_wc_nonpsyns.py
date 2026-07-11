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
from tensorflow.keras.constraints import Constraint, NonNeg
import h5py
import pandas as pd

import config
from utils.dataset import load_dataset_hdf5

class MinMaxCliper(tf.keras.constraints.Constraint):
    """
    Constraint: Clips weights element-wise to be within [min_val, max_val].

    Args:
        min_val: Minimum value. Can be scalar, numpy array, or tf.Tensor.
        max_val: Maximum value. Can be scalar, numpy array, or tf.Tensor.
    """
    def __init__(self, min_val, max_val):
        # Convert to tensors, preserve dtype (float32 by default)
        self.min_val = tf.constant(min_val, dtype=tf.float32)
        self.max_val = tf.constant(max_val, dtype=tf.float32)

    def __call__(self, w):
        """
        Clips the weights `w` element-wise to [min_val, max_val].

        broadcasting is supported: e.g., w=(10,5), min_val=(5,), max_val=1.0 → OK
        """
        min_broadcast = tf.broadcast_to(self.min_val, tf.shape(w))
        max_broadcast = tf.broadcast_to(self.max_val, tf.shape(w))
        return tf.clip_by_value(w, min_broadcast, max_broadcast)

    def get_config(self):
        # Required for serialization (e.g., model.save / model.from_config)
        return {
            'min_val': self.min_val.numpy().tolist() if self.min_val.ndim == 0 else tf.keras.backend.get_value(self.min_val).tolist(),
            'max_val': self.max_val.numpy().tolist() if self.max_val.ndim == 0 else tf.keras.backend.get_value(self.max_val).tolist(),
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

def _softplus(x):
    return tf.nn.softplus(x)


def _inv_softplus_np(y):
    """NumPy version for weight initialisation."""
    y = np.maximum(y, 1e-7)
    return np.log(np.expm1(y))


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

    def __init__(self, params, dt=0.1, batch_size=1, n_pre=None, **kwargs):
        super().__init__(**kwargs)
        self.dt = float(dt)
        self.units = config.N_POP_UNITS
        self.pre = n_pre if n_pre is not None else (config.N_POP_UNITS + config.N_INPUTS)
        self.post = config.N_POP_UNITS

        self.pconn = tf.constant(params['pconn'], dtype=tf.float32)
        # self.e_r = tf.constant(params['e_r'], dtype=tf.float32)

        # ei_sign = np.ones((self.pre, self.post), dtype=np.float32)
        # ei_sign[4:6, :] = -1.0
        ei_sign = np.sign(params['e_r'])
        self.ei_sign = tf.constant(ei_sign, dtype=tf.float32)


        wc_tau = np.array([20.0, 20.0, 20.0, 20.0, 10.0, 10.0], dtype=np.float32)
        wc_i_ext = np.ones( self.units, dtype=np.float32 ) + 5.0


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

        # ── Reparameterised weights ──────────────────────────────────
        # gsyn_max: g = softplus(θ), always > 0
        theta_gsyn = params['gsyn_max'] #_inv_softplus_np(params['gsyn_max'] * config.SYN_GSYN_INIT_SCALE)
        self._theta_gsyn = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(theta_gsyn),
            trainable=config.TRAIN_SYNAPSE_GMAX,
            name='theta_gsyn',
            constraint=NonNeg(),
        )

        # tau_1: τ_1 = exp(θ_1)
        theta_tau_1 = params['tau_1']  #np.log(np.maximum(params['tau_1'], 1e-7))
        self._theta_tau_1 = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(theta_tau_1),
            trainable=config.TRAIN_SYNAPSE_TAU_f,
            name='theta_tau_1',
            constraint=MinMaxCliper(min_val=2.0, max_val=15.0)
        )

        # tau_2: τ_2 = exp(θ_2)
        theta_tau_2 = params['tau_2']  # np.log(np.maximum(params['tau_2'], 1e-7))
        self._theta_tau_2 = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(theta_tau_2),
            trainable=config.TRAIN_SYNAPSE_TAU_d,
            name='theta_tau_2',
            constraint=MinMaxCliper(min_val=5.0, max_val=50.0)

        )

        self.state_size = [
            tf.TensorShape([batch_size, self.units]),
            tf.TensorShape([batch_size, self.pre, self.post]),
            tf.TensorShape([batch_size, self.pre, self.post]),
        ]

        self.output_size = self.units

    # ── Transform helpers (θ → physical) ─────────────────────────────
    def _get_gsyn(self):
        # return _softplus(self._theta_gsyn)
        return self._theta_gsyn

    def _get_tau_1(self):
        # return tf.exp(self._theta_tau_1)
        return self._theta_tau_1

    def _get_tau_2(self):
        # return tf.exp(self._theta_tau_2)
        return self._theta_tau_2

    def get_initial_state(self, batch_size=1):

        nu = tf.zeros([batch_size, self.units], dtype=tf.float32)
        g = tf.zeros([batch_size, self.pre, self.post], dtype=tf.float32)
        dg = tf.zeros([batch_size, self.pre, self.post], dtype=tf.float32)

        init_state = [nu, g, dg]

        return init_state

    def _naka_rushton(self, x):
        """S(x) = M * max(x,0)² / (σ² + max(x,0)²)"""
        x_pos = tf.maximum(x, 0.0)
        return self.WC_M * x_pos ** 2 / (self.WC_SIGMA ** 2 + x_pos ** 2)

    def call(self, inputs, states):
        nu, g, dg = states

        FRpre_unit = nu * self.dt # * 0.001
        FRpre_ext = inputs * self.dt # * 0.001


        if self.pre == self.units:
            FRpre = FRpre_ext
        else:
            FRpre = tf.concat([FRpre_unit, FRpre_ext], axis=1)



        FRpre_full = self.pconn[tf.newaxis, :, :] * FRpre[:, :, tf.newaxis]

        gsyn = self._get_gsyn()
        tau_1 = self._get_tau_1()
        tau_2 = self._get_tau_2()

        g_syn = gsyn * g
        I_syn = tf.reduce_sum(g_syn * self.ei_sign[tf.newaxis, :, :], axis=1)

        I_total = I_syn + self.I_ext[tf.newaxis, :]
        nu_new = nu + (self.dt / self.tau_pop[tf.newaxis, :]) * (
                -nu + self._naka_rushton(I_total))



        tau12 = tau_1 * tau_2
        t12_sum = tau_1 + tau_2

        dg_new = dg + self.dt * (FRpre_full - t12_sum * dg - g) / tau12
        g_new = g + self.dt * dg


        return nu_new, [nu_new, g_new, dg_new]


def gather_params(n_pre=None):
    if n_pre is None:
        n_pre = config.N_POP_UNITS + config.N_INPUTS
    n_units = config.N_POP_UNITS

    e_r = np.ones((n_pre, n_units), dtype=np.float32)
    e_r[4:6, :] = -1.0


    params = {
        'gsyn_max': np.random.uniform(0.0, 0.0001, size=(n_pre, n_units)).astype(np.float32),
        'tau_1': np.random.uniform(2.0, 6.0, size=(n_pre, n_units)).astype(np.float32),
        'tau_2': np.random.uniform(10.0, 30.0, size=(n_pre, n_units)).astype(np.float32),
        'pconn': np.ones((n_pre, n_units), dtype=np.float32),
        'e_r': e_r.astype(np.float32),

    }
    return params

def build_model(lr=1e-3, batch_size=1):
    inputs = Input(shape=(None, config.N_INPUTS), batch_size=batch_size)

    params = gather_params(n_pre=config.N_POP_UNITS + config.N_INPUTS)




    cell = WilsonCowanNetwork(params, dt=config.DT, batch_size=batch_size,
                               name='wc_layer')
    # 📊 Сохранение параметров в Excel
    # print("💾 Saving Wilson-Cowan parameters to Excel...")
    # with pd.ExcelWriter("wc_layer1_params.xlsx", engine="openpyxl") as writer:
    #     for name, value in params.items():
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

    x = RNN(cell, return_sequences=True, stateful=False, name='wc_rnn')(inputs)



    model = Model(inputs, x)

    def loss_with_reg(y_true, y_pred):
        L_mse = tf.keras.losses.MSLE(y_true, y_pred[..., :4])  #      #tf.keras.losses.MeanSquaredError()(y_true, y_pred[..., :4])  #      #tf.keras.losses.MeanSquaredError()(y_true, y_pred[..., :4])  #      #tf.keras.losses

        L_wta = config.WTA_WEIGHT * decorrelation_penalty(y_pred)
        L_sharp = config.LOSS_WEIGHT_SHARPENING * sharpening_loss(y_pred)
        L_ei = config.LOSS_WEIGHT_EI_BALANCE * ei_balance_loss(y_pred)

        return L_mse + L_wta + L_sharp + L_ei


    lr_schedule = lr
    # tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=lr,
    #     decay_steps=100000,
    #     alpha=0.01
    # )

    model.compile(
        optimizer=Adam(learning_rate=lr_schedule, clipnorm=1.0),
        loss=loss_with_reg,
        metrics=[R2Metric()],
    )
    return model



def load_all_batches(dataset_path):
    ds = load_dataset_hdf5(dataset_path)
    X = ds['X']
    Y = ds['Y']
    print(f"  X: {X.shape}, Y: {Y.shape}, {X.nbytes / 1e6:.1f} MB")
    return X, Y


def train(dataset_path=None, n_epochs=None, learning_rate=None,
         seed=None, resume=None, n_layers=1, batch_size=None,
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

    print(f"Loading dataset from {ds_path}...")
    X, Y = load_all_batches(ds_path)

    if X.shape[0] > 10000:
        n_val = max(1, int(len(X) * val_split))
        X_val, Y_val = X[-n_val:], Y[-n_val:]
        X_train, Y_train = X[:-n_val], Y[:-n_val]
    else:
        X_val, Y_val = X, Y
        X_train, Y_train = X, Y

    print(f"  Train: {X_train.shape[0]} batches, Val: {X_val.shape[0]} batches")

    print(f"Building Wilson-Cowan model ({n_layers} layers)...")
    model = build_model(lr=lr, batch_size=batch_size)
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
