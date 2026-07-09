"""Parameter matrices and custom layer for Wilson-Cowan sub-models (Phase 1).

Each sub-model has 3 units (Border_X, Basket, Axo) and 24 input channels:
21 real (d_far, d_near, speed, HDx18) + 3 teacher channels carrying the
ideal targets of the OTHER three pyramids.

Layer: WilsonCowanSubNetwork -- same dynamics as
train_wc_nonpsyns.WilsonCowanNetwork but with parametric units/post
dimensions (3, 3) instead of hard-coded (6, 6). Variable names match the
full model so weight files are interchangeable for the variables that
exist (I_ext, theta_gsyn, theta_tau_1, theta_tau_2, tau_pop).
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

import config

from utils.submodel import augment_with_teachers, extract_target_for


WC_TAU_FULL = np.array([12.0, 12.0, 12.0, 12.0, 10.0, 10.0], dtype=np.float32)

N_SUB_INPUTS = config.N_INPUTS + 3


def _softplus(x):
    return tf.nn.softplus(x)


def _inv_softplus_np(y):
    y = np.maximum(y, 1e-7)
    return np.log(np.expm1(y))


class WilsonCowanSubNetwork(Layer):
    """Wilson-Cowan rate model + TM synapses with parametric units/post dims."""

    WC_M = 100.0
    WC_SIGMA = 5.0

    def __init__(self, params, dt=0.1, batch_size=1, n_pre=None,
                 n_units=3, n_post=3, **kwargs):
        super().__init__(**kwargs)
        self.dt = float(dt)
        self.units = n_units
        self.pre = n_pre if n_pre is not None else n_units
        self.post = n_post

        self.pconn = tf.constant(params['pconn'], dtype=tf.float32)
        ei_sign = np.sign(params['e_r'])
        self.ei_sign = tf.constant(ei_sign, dtype=tf.float32)

        wc_tau = WC_TAU_FULL[:n_units]
        wc_i_ext = np.ones(n_units, dtype=np.float32) + 5.0

        self.tau_pop = self.add_weight(
            shape=(n_units,),
            initializer=tf.constant_initializer(wc_tau),
            trainable=False,
            name='tau_pop',
        )

        self.I_ext = self.add_weight(
            shape=(n_units,),
            initializer=tf.constant_initializer(wc_i_ext),
            trainable=config.TRAIN_POP_IEXT,
            name='I_ext',
        )

        theta_gsyn = _inv_softplus_np(
            params['gsyn_max'] * config.SYN_GSYN_INIT_SCALE)
        self._theta_gsyn = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(theta_gsyn),
            trainable=config.TRAIN_SYNAPSE_GMAX,
            name='theta_gsyn',
        )

        theta_tau_1 = np.log(np.maximum(params['tau_1'], 1e-7))
        self._theta_tau_1 = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(theta_tau_1),
            trainable=config.TRAIN_SYNAPSE_TAU_f,
            name='theta_tau_1',
        )

        theta_tau_2 = np.log(np.maximum(params['tau_2'], 1e-7))
        self._theta_tau_2 = self.add_weight(
            shape=(self.pre, self.post),
            initializer=tf.constant_initializer(theta_tau_2),
            trainable=config.TRAIN_SYNAPSE_TAU_d,
            name='theta_tau_2',
        )

        self.state_size = [
            tf.TensorShape([batch_size, self.units]),
            tf.TensorShape([batch_size, self.pre, self.post]),
            tf.TensorShape([batch_size, self.pre, self.post]),
        ]
        self.output_size = self.units

    def _get_gsyn(self):
        return _softplus(self._theta_gsyn)

    def _get_tau_1(self):
        return tf.exp(self._theta_tau_1)

    def _get_tau_2(self):
        return tf.exp(self._theta_tau_2)

    def get_initial_state(self, batch_size=1):
        nu = tf.zeros([batch_size, self.units], dtype=tf.float32)
        g = tf.zeros([batch_size, self.pre, self.post], dtype=tf.float32)
        dg = tf.zeros([batch_size, self.pre, self.post], dtype=tf.float32)
        return [nu, g, dg]

    def _naka_rushton(self, x):
        x_pos = tf.maximum(x, 0.0)
        return self.WC_M * x_pos ** 2 / (self.WC_SIGMA ** 2 + x_pos ** 2)

    def call(self, inputs, states):
        nu, g, dg = states

        FRpre_unit = nu * self.dt
        FRpre_ext = inputs * self.dt

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


def build_submodel_wc_params(X_idx, teacher_gsyn=1.0, rng_seed=None):
    """Build (gsyn_max, tau_1, tau_2, pconn, e_r, I_ext) for a 3-unit WC sub-model.

    Sub-model unit order: [Border_X, Basket, Axo]  (pyramidal at index 0)
    Sub-model pre layout: [Border_X, Basket, Axo, 21 real inputs, 3 teacher]
                          = 27 rows.
    Sub-model post layout: [Border_X, Basket, Axo] = 3 cols.

    The teacher rows carry the OTHER pyramids' ideal targets (in increasing-
    index order). They are initialised with a HIGHER gsyn_max so that the
    teacher signal is meaningful from step 0.
    """
    assert X_idx in (0, 1, 2, 3), f"X_idx must be 0..3, got {X_idx}"

    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed + X_idx)
    else:
        rng = np.random

    other_pyramids = [i for i in range(4) if i != X_idx]
    n_pre = 3 + config.N_INPUTS + len(other_pyramids)
    n_post = 3

    rec_end = 3
    inp_end = rec_end + config.N_INPUTS
    teacher_off = inp_end

    gsyn_max = rng.uniform(0.0, 0.0001, size=(n_pre, n_post)).astype(np.float32)
    gsyn_max[teacher_off:, :] = teacher_gsyn
    tau_1 = rng.uniform(2.0, 6.0, size=(n_pre, n_post)).astype(np.float32)
    tau_2 = rng.uniform(10.0, 30.0, size=(n_pre, n_post)).astype(np.float32)

    pconn = np.ones((n_pre, n_post), dtype=np.float32)

    e_r = np.ones((n_pre, n_post), dtype=np.float32)
    e_r[1, :] = -1.0
    e_r[2, :] = -1.0

    I_ext = np.ones(n_post, dtype=np.float32) + 5.0

    return {
        "gsyn_max": gsyn_max,
        "tau_1": tau_1,
        "tau_2": tau_2,
        "pconn": pconn,
        "e_r": e_r,
        "I_ext": I_ext,
    }