"""Custom input generators for border cell simulation."""

import tensorflow as tf
import numpy as np
from neuraltide.inputs import BaseInputGenerator
import config


class DistanceFarGenerator(BaseInputGenerator):
    """d_far: fires MORE when FAR from nearest wall.
    r(t) = alpha_far * d_min(t). d_min from extra_inputs col 0."""

    def __init__(self, params=None, dt=None, **kwargs):
        if params is None:
            params = {'_': 0.0}
        if dt is None:
            dt = config.SIM_DT
        super().__init__(params=params, dt=dt, **kwargs)
        self._alpha_far = tf.constant(config.ALPHA_FAR,
                                       dtype=tf.keras.backend.floatx())

    def call(self, t, extra_inputs=None, **call_kwargs):
        d_min = tf.reshape(extra_inputs[:, 0], [-1])
        rate = self._alpha_far * d_min
        return rate


class DistanceNearGenerator(BaseInputGenerator):
    """d_near: fires MORE when NEAR a wall.
    r(t) = alpha_near * (D_max - d_min(t)).
    alpha_near > 0 means rate = max at d_min=0, declines linearly to 0 at d_min=D_max."""

    def __init__(self, params=None, dt=None, **kwargs):
        if params is None:
            params = {'_': 0.0}
        if dt is None:
            dt = config.SIM_DT
        super().__init__(params=params, dt=dt, **kwargs)
        self._alpha_near = tf.constant(config.ALPHA_NEAR,
                                        dtype=tf.keras.backend.floatx())
        self._d_max = tf.constant(config.D_MAX, dtype=tf.keras.backend.floatx())

    def call(self, t, extra_inputs=None, **call_kwargs):
        d_min = tf.reshape(extra_inputs[:, 0], [-1])
        rate = self._alpha_near * (self._d_max - d_min)
        return tf.maximum(rate, 0.0)


class SpeedGenerator(BaseInputGenerator):
    """Speed cell: r(t) = beta_0 + beta_1 * |v(t)|.
    Speed from extra_inputs col 1."""

    def __init__(self, params=None, dt=None, **kwargs):
        if params is None:
            params = {'_': 0.0}
        if dt is None:
            dt = config.SIM_DT
        super().__init__(params=params, dt=dt, **kwargs)
        self._beta_0 = tf.constant(config.BETA_0, dtype=tf.keras.backend.floatx())
        self._beta_1 = tf.constant(config.BETA_1, dtype=tf.keras.backend.floatx())

    def call(self, t, extra_inputs=None, **call_kwargs):
        speed = tf.reshape(extra_inputs[:, 1], [-1])
        rate = self._beta_0 + self._beta_1 * speed
        return rate


class HDPopVecGenerator(BaseInputGenerator):
    """Head direction population vector (2D, n_units=2).
    HD_vec = Σ r_HD_i * [cos θ_pref,i, sin θ_pref,i].
    Receives cos(θ) from extra_inputs col 2, sin(θ) from col 3."""

    def __init__(self, params=None, dt=None, **kwargs):
        if params is None:
            params = {'_': [0.0, 0.0]}
        if dt is None:
            dt = config.SIM_DT
        super().__init__(params=params, dt=dt, **kwargs)
        dtype = tf.keras.backend.floatx()
        self._f_max = tf.cast(config.F_MAX_HD, dtype=dtype)
        self._kappa = tf.cast(config.KAPPA_HD, dtype=dtype)
        self._theta_pref_rad = tf.constant(
            [np.deg2rad(th) for th in config.THETA_PREF], dtype=dtype)
        self._cos_pref = tf.cos(self._theta_pref_rad)
        self._sin_pref = tf.sin(self._theta_pref_rad)

    def call(self, t, extra_inputs=None, **call_kwargs):
        cos_hd = tf.reshape(extra_inputs[:, 2], [-1, 1])
        sin_hd = tf.reshape(extra_inputs[:, 3], [-1, 1])
        theta = tf.math.atan2(sin_hd, cos_hd)
        r_hd = self._f_max * tf.exp(
            self._kappa * (tf.cos(theta - self._theta_pref_rad) - 1.0))
        hd_x = tf.reduce_sum(r_hd * self._cos_pref, axis=1, keepdims=True)
        hd_y = tf.reduce_sum(r_hd * self._sin_pref, axis=1, keepdims=True)
        return tf.concat([hd_x, hd_y], axis=1)
