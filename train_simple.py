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

    def __init__(self, params, dt_dim=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dt_dim = float(dt_dim)
        self.units = 6
        self.pre = config.N_POP_UNITS + config.N_INPUTS
        self.post = config.N_POP_UNITS
        self.PI = float(np.pi)
        self.v_max = 10.0

        self.alpha = tf.constant(params['alpha'], dtype=tf.float32)
        self.a = tf.constant(params['a'], dtype=tf.float32)
        self.b = tf.constant(params['b'], dtype=tf.float32)
        self.w_jump = tf.constant(params['w_jump'], dtype=tf.float32)
        self.tau_pop = tf.constant(params['tau_pop'], dtype=tf.float32)
        self.pconn = tf.constant(params['pconn'], dtype=tf.float32)
        self.e_r = tf.constant(params['e_r'], dtype=tf.float32)

        self.Delta_I = self.add_weight(
            shape=(self.units,),
            initializer=tf.constant_initializer(params['Delta_I']),
            trainable=config.TRAIN_POP_DELTA_I,
            constraint=MinMax(0.0001, 0.5),
            name='Delta_I',
        )
        self.I_ext = self.add_weight(
            shape=(self.units,),
            initializer=tf.constant_initializer(params['I_ext']),
            trainable=config.TRAIN_POP_IEXT,
            name='I_ext',
        )
        self.gsyn_max = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(params['gsyn_max']),
            trainable=config.TRAIN_SYNAPSE_GMAX,
            constraint=tf.keras.constraints.NonNeg(),
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
            constraint=MinMax(0.04, 0.4),
            name='Uinc',
        )

        self.state_size = [
            tf.TensorShape([1, self.units]),
            tf.TensorShape([1, self.units]),
            tf.TensorShape([1, self.units]),
            tf.TensorShape([1, self.pre, self.post]),
            tf.TensorShape([1, self.pre, self.post]),
            tf.TensorShape([1, self.pre, self.post]),
        ]
        self.output_size = self.units

    def get_initial_state(self, batch_size=1):
        return [
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.ones([batch_size, self.pre, self.post], dtype=tf.float32),
            tf.zeros([batch_size, self.pre, self.post], dtype=tf.float32),
            tf.zeros([batch_size, self.pre, self.post], dtype=tf.float32),
        ]

    def _deriv(self, r, v, w, A):
        g_syn = self.gsyn_max * A
        g_syn_tot = tf.reduce_sum(g_syn, axis=1)
        I_syn = tf.reduce_sum(g_syn * (self.e_r - v), axis=1)
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

        #dt_per_tau = self.dt_dim / self.tau_pop
        FRpre_unit = r   #* dt_per_tau
        FRpre_ext = ext * 0.001 * self.dt_dim
        FRpre = tf.concat([FRpre_unit, FRpre_ext], axis=1)
        FRpre_full = self.pconn[tf.newaxis, :, :] * FRpre[:, :, tf.newaxis]

        exp_d = tf.exp(-self.dt_dim / self.tau_d)
        exp_r = tf.exp(-self.dt_dim / self.tau_r)
        exp_f = tf.exp(-self.dt_dim / self.tau_f)
        tau1r = tf.where(self.tau_d != self.tau_r,
                         self.tau_d / (self.tau_d - self.tau_r),
                         1e-13)
        exp_d_b = exp_d[tf.newaxis, :, :]
        exp_r_b = exp_r[tf.newaxis, :, :]
        exp_f_b = exp_f[tf.newaxis, :, :]
        tau1r_b = tau1r[tf.newaxis, :, :]
        Uinc_b = self.Uinc[tf.newaxis, :, :]

        a_ = A * exp_d_b
        r_ = 1.0 + (R - 1.0 + tau1r_b * A) * exp_r_b - tau1r_b * A
        u_ = U * exp_f_b
        released = U * r_ * FRpre_full
        U_new = u_ + Uinc_b * (1.0 - u_) * FRpre_full
        A_new = a_ + released
        R_new = r_ - released

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
    return {
        'alpha': np.asarray(pop['alpha'], dtype=np.float32),
        'a': np.asarray(pop['a'], dtype=np.float32),
        'b': np.asarray(pop['b'], dtype=np.float32),
        'w_jump': np.asarray(pop['w_jump'], dtype=np.float32),
        'tau_pop': np.asarray(pop['tau_pop'], dtype=np.float32),
        'I_ext': np.asarray(pop['I_ext'], dtype=np.float32),
        'Delta_I': np.asarray(pop['Delta_I'], dtype=np.float32),
        'gsyn_max': gsyn.astype(np.float32),
        'tau_d': tau_d.astype(np.float32),
        'tau_r': tau_r.astype(np.float32),
        'tau_f': tau_f.astype(np.float32),
        'Uinc': Uinc.astype(np.float32),
        'pconn': pconn.astype(np.float32),
        'e_r': e_r.astype(np.float32),
    }


def build_model():
    params = gather_params()
    inputs = Input(shape=(None, config.N_INPUTS), batch_size=1)
    cell = BorderMeanFieldNetwork(params, dt_dim=config.DT)
    rnn = RNN(cell, return_sequences=True, stateful=True, name='border_rnn')
    out = rnn(inputs)
    model = Model(inputs, out)

    def msle_first4(y_true, y_pred):
        return tf.keras.losses.MeanSquaredLogarithmicError()(
            y_true, y_pred[..., :4])

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE, clipvalue=10.0),
        loss=msle_first4,
    )
    return model


def load_pretrained(model, path):
    """Assign saved values to matching trainable variables (best-effort)."""
    if not os.path.exists(path):
        print(f"  No pretrained file at {path}, starting from scratch.")
        return 0
    with h5py.File(path, 'r') as f:
        if 'parameters' not in f:
            print(f"  No 'parameters' group in {path}, starting from scratch.")
            return 0
        grp = f['parameters']
        var_map = {}
        for v in model.trainable_variables:
            for cand in (v.name,
                         v.name.replace(':', '_').replace('/', '_'),
                         v.name.split('/')[-1].split(':')[0]):
                var_map.setdefault(cand, v)
        loaded = 0
        for saved_name in grp.keys():
            if saved_name not in var_map:
                print(f"  WARNING: '{saved_name}' in file but no matching variable in model")
                continue
            v = var_map[saved_name]
            saved_shape = grp[saved_name].shape
            if tuple(saved_shape) != tuple(v.shape):
                print(f"  WARNING: '{saved_name}' shape mismatch: "
                      f"saved={saved_shape}, model={tuple(v.shape)}")
                continue
            v.assign(grp[saved_name][:])
            loaded += 1
            print(f"  Loaded {saved_name}: shape={tuple(v.shape)}")
    print(f"  Loaded {loaded} variable(s) from {path}.")
    return loaded


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
    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
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
          batches_per_epoch=None, seed=None, resume=None):
    ds_path = dataset_path or os.path.join(
        os.path.dirname(config.TRAJECTORY_HDF5), 'dataset.h5')
    n_epochs = n_epochs or config.N_EPOCHS
    lr = learning_rate or config.LEARNING_RATE
    batches_per_epoch = batches_per_epoch or config.N_BATCHES_PER_EPOCH
    if seed is not None:
        config.RANDOM_SEED = seed
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print(f"Loading dataset from {ds_path}...")
    X, Y = load_all_batches(ds_path)
    n_batches = X.shape[0]
    if batches_per_epoch > n_batches:
        raise ValueError(
            f"batches_per_epoch ({batches_per_epoch}) > n_batches ({n_batches})")

    print("Building model...")
    model = build_model()
    n_vars = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"  Trainable parameters: {n_vars}")
    print(f"  Trainable variables: {[v.name for v in model.trainable_variables]}")

    if resume:
        print(f"Resuming from {resume}...")
        load_pretrained(model, resume)

    loss_history = []
    best_loss = float('inf')
    print(f"Training: {n_epochs} epochs x {batches_per_epoch} batches/epoch "
          f"(out of {n_batches})...")
    t_start = time.time()
    for epoch in range(n_epochs):
        epoch_t0 = time.time()
        idx = np.random.choice(n_batches, size=batches_per_epoch, replace=False)
        epoch_loss = 0.0
        n_finite = 0
        for i in idx:
            loss = model.train_on_batch(X[i], Y[i])
            if np.isfinite(loss):
                epoch_loss += float(loss)
                n_finite += 1
        avg_loss = epoch_loss / max(n_finite, 1)
        loss_history.append(float(avg_loss))
        if avg_loss < best_loss:
            best_loss = avg_loss
        epoch_dt = time.time() - epoch_t0
        if (epoch + 1) % max(1, n_epochs // 20) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs} | loss={avg_loss:.6f} | "
                  f"best={best_loss:.6f} | finite={n_finite}/{batches_per_epoch} "
                  f"| dt={epoch_dt:.1f}s")
    total_dt = time.time() - t_start
    print(f"Training done in {total_dt/60:.1f} min.")
    save_training_results(loss_history, model)
    return loss_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batches-per-epoch', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a previous results/training.h5 to load '
                             'trained weights from before training continues.')
    args = parser.parse_args()
    train(args.dataset, args.epochs, args.lr, args.batches_per_epoch,
          args.seed, args.resume)


if __name__ == '__main__':
    main()
