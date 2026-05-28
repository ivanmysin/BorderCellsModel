"""Custom input generators for border cell simulation."""

import tensorflow as tf
import numpy as np
from neuraltide.inputs import BaseInputGenerator
import config


class BaseDistanceGenerator(BaseInputGenerator):
    """Base class for distance-to-wall generators.

    Computes d_min (distance to nearest wall) from x,y coordinates
    and arena size. Subclasses override call() to define rate = f(d_min).
    """

    def __init__(self, params=None, dt=None, arena_cm=None, **kwargs):
        if params is None:
            params = {'_': 0.0}
        if dt is None:
            dt = config.SIM_DT
        super().__init__(params=params, dt=dt, **kwargs)
        self._arena_cm_tf = tf.constant(
            arena_cm if arena_cm is not None else config.ARENA_CM,
            dtype=tf.keras.backend.floatx(),
        )

    def _compute_d_min(self, x, y):
        d_N = self._arena_cm_tf - y
        d_S = y
        d_E = self._arena_cm_tf - x
        d_W = x
        return tf.minimum(tf.minimum(d_N, d_S), tf.minimum(d_E, d_W))


class DistanceFarGenerator(BaseDistanceGenerator):
    """d_far: fires MORE when FAR from nearest wall.
    r(t) = alpha_far * d_min(t). d_min computed from extra_inputs x,y."""

    def __init__(self, params=None, dt=None, arena_cm=None, **kwargs):
        super().__init__(params=params, dt=dt, arena_cm=arena_cm, **kwargs)
        self._alpha_far = tf.constant(config.ALPHA_FAR,
                                       dtype=tf.keras.backend.floatx())

    def call(self, t, extra_inputs=None, **call_kwargs):
        x = tf.reshape(extra_inputs[:, 0], [-1])
        y = tf.reshape(extra_inputs[:, 1], [-1])
        d_min = self._compute_d_min(x, y)
        rate = self._alpha_far * d_min
        return rate


class DistanceNearGenerator(BaseDistanceGenerator):
    """d_near: fires MORE when NEAR a wall.
    r(t) = alpha_near * (D_max - d_min(t)).
    alpha_near > 0 means rate = max at d_min=0, declines linearly to 0 at d_min=D_max."""

    def __init__(self, params=None, dt=None, arena_cm=None, **kwargs):
        super().__init__(params=params, dt=dt, arena_cm=arena_cm, **kwargs)
        self._alpha_near = tf.constant(config.ALPHA_NEAR,
                                        dtype=tf.keras.backend.floatx())
        self._d_max = tf.constant(config.D_MAX, dtype=tf.keras.backend.floatx())

    def call(self, t, extra_inputs=None, **call_kwargs):
        x = tf.reshape(extra_inputs[:, 0], [-1])
        y = tf.reshape(extra_inputs[:, 1], [-1])
        d_min = self._compute_d_min(x, y)
        rate = self._alpha_near * (self._d_max - d_min)
        return tf.maximum(rate, 0.0)


class SpeedGenerator(BaseInputGenerator):
    """Speed cell: r(t) = beta_0 + beta_1 * |v(t)|.
    Speed components from extra_inputs cols 2 (vx), 3 (vy)."""

    def __init__(self, params=None, dt=None, **kwargs):
        if params is None:
            params = {'_': 0.0}
        if dt is None:
            dt = config.SIM_DT
        super().__init__(params=params, dt=dt, **kwargs)
        self._beta_0 = tf.constant(config.BETA_0, dtype=tf.keras.backend.floatx())
        self._beta_1 = tf.constant(config.BETA_1, dtype=tf.keras.backend.floatx())

    def call(self, t, extra_inputs=None, **call_kwargs):
        vx = tf.reshape(extra_inputs[:, 2], [-1])
        vy = tf.reshape(extra_inputs[:, 3], [-1])
        speed = tf.sqrt(vx**2 + vy**2)
        rate = self._beta_0 + self._beta_1 * speed
        return rate


class HeadDirectionGenerator(BaseInputGenerator):
    """Head direction population (n_units=N_HD=18).
    Each unit is an HD cell with von Mises tuning.
    Receives vx from extra_inputs col 2, vy from col 3 — computes θ = atan2(vy, vx).
    Returns firing rates for all N_HD cells, shape [batch, N_HD]."""

    def __init__(self, params=None, dt=None, **kwargs):
        if params is None:
            params = {'_': [0.0] * config.N_HD}
        if dt is None:
            dt = config.SIM_DT
        super().__init__(params=params, dt=dt, **kwargs)
        dtype = tf.keras.backend.floatx()
        self._kappa = tf.cast(config.KAPPA_HD, dtype=dtype)
        self._f_max = tf.cast(config.F_MAX_HD, dtype=dtype) / tf.exp(self._kappa)

        self._theta_pref_rad = tf.constant(
            [np.deg2rad(th) for th in config.THETA_PREF], dtype=dtype)

    def call(self, t, extra_inputs=None, **call_kwargs):
        vx = tf.reshape(extra_inputs[:, 2], [-1, 1])
        vy = tf.reshape(extra_inputs[:, 3], [-1, 1])
        theta = tf.math.atan2(vy, vx)
        rate_hd = self._f_max * tf.exp(
            self._kappa * tf.cos(theta - self._theta_pref_rad))
        return rate_hd
