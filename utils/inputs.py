"""Custom input generators for border cell simulation."""

import tensorflow as tf
import numpy as np
# from neuraltide.inputs import BaseInputGenerator
import config

#
# class BaseDistanceGenerator(BaseInputGenerator):
#     """Base class for distance-to-wall generators.
#
#     Computes d_min (distance to nearest wall) from x,y coordinates
#     and arena size. Subclasses override call() to define rate = f(d_min).
#     """
#
#     def __init__(self, params=None, dt=None, arena_cm=None, **kwargs):
#         if params is None:
#             params = {'_': 0.0}
#         if dt is None:
#             dt = config.SIM_DT
#         super().__init__(params=params, dt=dt, **kwargs)
#         self._arena_cm_tf = tf.constant(
#             arena_cm if arena_cm is not None else config.ARENA_CM,
#             dtype=tf.keras.backend.floatx(),
#         )
#
#     def _compute_d_min(self, x, y):
#         d_N = self._arena_cm_tf - y
#         d_S = y
#         d_E = self._arena_cm_tf - x
#         d_W = x
#         return tf.minimum(tf.minimum(d_N, d_S), tf.minimum(d_E, d_W))
#
#
# class DistanceFarGenerator(BaseDistanceGenerator):
#     """d_far: fires MORE when FAR from nearest wall.
#     r(t) = alpha_far * d_min(t). d_min computed from extra_inputs x,y."""
#
#     def __init__(self, params=None, dt=None, arena_cm=None, **kwargs):
#         super().__init__(params=params, dt=dt, arena_cm=arena_cm, **kwargs)
#         self._alpha_far = tf.constant(config.ALPHA_FAR,
#                                        dtype=tf.keras.backend.floatx())
#
#     def call(self, t, extra_inputs=None, **call_kwargs):
#         x = tf.reshape(extra_inputs[:, 0], [-1])
#         y = tf.reshape(extra_inputs[:, 1], [-1])
#         d_min = self._compute_d_min(x, y)
#         rate = self._alpha_far * d_min
#         return rate
#
#
# class DistanceNearGenerator(BaseDistanceGenerator):
#     """d_near: fires MORE when NEAR a wall.
#     r(t) = alpha_near * (D_max - d_min(t)).
#     alpha_near > 0 means rate = max at d_min=0, declines linearly to 0 at d_min=D_max."""
#
#     def __init__(self, params=None, dt=None, arena_cm=None, **kwargs):
#         super().__init__(params=params, dt=dt, arena_cm=arena_cm, **kwargs)
#         self._alpha_near = tf.constant(config.ALPHA_NEAR,
#                                         dtype=tf.keras.backend.floatx())
#         self._d_max = tf.constant(config.D_MAX, dtype=tf.keras.backend.floatx())
#
#     def call(self, t, extra_inputs=None, **call_kwargs):
#         x = tf.reshape(extra_inputs[:, 0], [-1])
#         y = tf.reshape(extra_inputs[:, 1], [-1])
#         d_min = self._compute_d_min(x, y)
#         rate = self._alpha_near * (self._d_max - d_min)
#         return tf.maximum(rate, 0.0)
#
#
# class SpeedGenerator(BaseInputGenerator):
#     """Speed cell: r(t) = beta_0 + beta_1 * |v(t)|.
#     Speed components from extra_inputs cols 2 (vx), 3 (vy)."""
#
#     def __init__(self, params=None, dt=None, **kwargs):
#         if params is None:
#             params = {'_': 0.0}
#         if dt is None:
#             dt = config.SIM_DT
#         super().__init__(params=params, dt=dt, **kwargs)
#         self._beta_0 = tf.constant(config.BETA_0, dtype=tf.keras.backend.floatx())
#         self._beta_1 = tf.constant(config.BETA_1, dtype=tf.keras.backend.floatx())
#
#     def call(self, t, extra_inputs=None, **call_kwargs):
#         vx = tf.reshape(extra_inputs[:, 2], [-1])
#         vy = tf.reshape(extra_inputs[:, 3], [-1])
#         speed = tf.sqrt(vx**2 + vy**2)
#         rate = self._beta_0 + self._beta_1 * speed
#         return rate
#
#
# class HeadDirectionGenerator(BaseInputGenerator):
#     """Head direction population (n_units=N_HD=18).
#     Each unit is an HD cell with von Mises tuning.
#     Receives vx from extra_inputs col 2, vy from col 3 — computes θ = atan2(vy, vx).
#     Returns firing rates for all N_HD cells, shape [batch, N_HD]."""
#
#     def __init__(self, params=None, dt=None, **kwargs):
#         if params is None:
#             params = {'_': [0.0] * config.N_HD}
#         if dt is None:
#             dt = config.SIM_DT
#         super().__init__(params=params, dt=dt, **kwargs)
#         dtype = tf.keras.backend.floatx()
#         self._kappa = tf.cast(config.KAPPA_HD, dtype=dtype)
#         self._f_max = tf.cast(config.F_MAX_HD, dtype=dtype) / tf.exp(self._kappa)
#
#         self._theta_pref_rad = tf.constant(
#             [np.deg2rad(th) for th in config.THETA_PREF], dtype=dtype)
#
#     def call(self, t, extra_inputs=None, **call_kwargs):
#         vx = tf.reshape(extra_inputs[:, 2], [-1, 1])
#         vy = tf.reshape(extra_inputs[:, 3], [-1, 1])
#         theta = tf.math.atan2(vy, vx)
#         rate_hd = self._f_max * tf.exp(
#             self._kappa * tf.cos(theta - self._theta_pref_rad))
#         return rate_hd


def precompute_inputs(traj: dict) -> np.ndarray:
    """Precompute all input channels from trajectory data.

    Uses the same formulas as the generator classes but implemented as
    vectorized NumPy for efficient batch precomputation.

    Channel layout (N_INPUTS=21, see config.N_INPUTS):
        0:        d_far   (alpha_far * d_min)
        1:        d_near  (alpha_near * (D_max - d_min), clipped >= 0)
        2:        speed   (beta_0 + beta_1 * |v|)
        3..10:    CB×8    (center-bearing: von Mises on egocentric bearing
                            to geometric center; Long et al. 2025)
        11..18:   CD×HD×8 (conjunctive CD×HD: HD_tuning * CD_norm;
                            replaces previous 18 allocentric HD cells)
        19:       cd_far  (alpha_cd_far * CD; positive CD slope)
        20:       cd_near (alpha_cd_near * (CD_max - CD), clipped >= 0;
                            negative CD slope)

    Args:
        traj: trajectory dict with keys x, y, vx, vy, speed, head_direction, etc.

    Returns:
        np.ndarray of shape [n_steps, N_INPUTS] with input rates in Hz.
    """
    n_steps = traj['x'].shape[1]
    x = traj['x']
    y = traj['y']
    speed = traj['speed']
    hd = traj['head_direction']

    arena = config.ARENA_CM
    cx, cy = arena / 2.0, arena / 2.0  # geometric center of the square arena

    # Distance to nearest wall (same as BaseDistanceGenerator._compute_d_min)
    d_N = arena - y
    d_S = y
    d_E = arena - x
    d_W = x
    d_min = np.minimum(np.minimum(d_N, d_S), np.minimum(d_E, d_W))

    # d_far: same formula as DistanceFarGenerator
    d_far = config.ALPHA_FAR * d_min

    # d_near: same formula as DistanceNearGenerator
    d_near = config.ALPHA_NEAR * (config.D_MAX - d_min)
    d_near = np.maximum(d_near, 0.0)

    # speed: same formula as SpeedGenerator
    speed_rate = config.BETA_0 + config.BETA_1 * speed

    # ------------------------------------------------------------
    # Center-bearing (CB): egocentric bearing to geometric center
    # ------------------------------------------------------------
    # θ_to_C = allocentric direction from animal to center
    # CB = θ_to_C - θ_HD  (wrapped to (-π, π])
    # All formulas in the article (Long et al. 2025, Methods,
    # "Identification of center-bearing cells"):
    #   CB = atan2(sin(θ_to_C - θ_HD), cos(θ_to_C - θ_HD))
    theta_to_center = np.arctan2(cy - y, cx - x)
    delta = theta_to_center - hd
    CB = np.arctan2(np.sin(delta), np.cos(delta))  # [batch, n_steps]

    theta_pref_cb_rad = np.deg2rad(config.THETA_PREF_CB)  # [N_CB]
    f_max_cb = config.F_MAX_CB / np.exp(config.KAPPA_CB)
    cb_rates = f_max_cb * np.exp(
        config.KAPPA_CB * np.cos(CB[:, :, np.newaxis] - theta_pref_cb_rad)
    )  # [batch, n_steps, N_CB]

    # ------------------------------------------------------------
    # Center-distance (CD): distance from animal to geometric center
    # ------------------------------------------------------------
    CD = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)  # [batch, n_steps]

    # cd_far: positive slope (fires more when far from center)
    cd_far = config.ALPHA_CD_FAR * CD
    # cd_near: negative slope (fires more when near center)
    cd_near = config.ALPHA_CD_NEAR * (config.CD_MAX - CD)
    cd_near = np.maximum(cd_near, 0.0)

    # ------------------------------------------------------------
    # CD×HD: conjunctive (allocentric head direction) × (positive CD slope)
    # ------------------------------------------------------------
    # 8 cells, each with a preferred allocentric HD and a CD modulation
    # r_k(HD, CD) = h_k(HD) * g(CD), where:
    #   h_k(HD) = (F_MAX_HD / exp(κ)) * exp(κ * cos(θ_HD - θ_pref_k))   # von Mises
    #   g(CD)   = sqrt(clip(CD / CD_max, 0, 1))                          # sqrt ramp
    # The sqrt transform amplifies the position signal in the mid-arena
    # zone (where the agent spends ~40% of the time per thigmotaxis=0.6)
    # without losing the linear information near the edges.
    theta_pref_hd_rad = np.deg2rad(config.THETA_PREF_HD)  # [N_HD]
    f_max_hd = config.F_MAX_HD / np.exp(config.KAPPA_HD)
    hd_tuning = f_max_hd * np.exp(
        config.KAPPA_HD * np.cos(hd[:, :, np.newaxis] - theta_pref_hd_rad)
    )  # [batch, n_steps, N_HD]
    cd_norm = np.sqrt(np.clip(CD / config.CD_MAX, 0.0, 1.0))[:, :, np.newaxis]
    cdhd_rates = hd_tuning * cd_norm

    d_far = d_far[:, :, np.newaxis]
    d_near = d_near[:, :, np.newaxis]
    speed_rate = speed_rate[:, :, np.newaxis]
    cd_far = cd_far[:, :, np.newaxis]
    cd_near = cd_near[:, :, np.newaxis]

    return np.concat([
        d_far,        # 1
        d_near,       # 1
        speed_rate,   # 1
        cb_rates,     # 8
        cdhd_rates,   # 8
        cd_far,       # 1
        cd_near,      # 1
    ], axis=-1).astype(np.float32)
