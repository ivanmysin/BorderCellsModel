"""Tests for input generators and precompute_inputs."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.inputs import precompute_inputs


@pytest.fixture
def sample_trajectory():
    """Create a simple synthetic trajectory for testing.

    All arrays are 2D with shape (1, n_steps) — precompute_inputs()
    assumes this layout (matches generate_trajectory.py output).
    """
    n = 1000
    rng = np.random.RandomState(42)
    x = rng.uniform(5, config.ARENA_CM - 5, n)
    y = rng.uniform(5, config.ARENA_CM - 5, n)
    speed = rng.uniform(5, 40, n)
    hd = rng.uniform(-np.pi, np.pi, n)
    vx = speed * np.cos(hd)
    vy = speed * np.sin(hd)
    d_N = config.ARENA_CM - y
    d_S = y
    d_E = config.ARENA_CM - x
    d_W = x
    d_min = np.minimum(np.minimum(d_N, d_S), np.minimum(d_E, d_W))
    t = np.arange(n) * config.TRAJECTORY_DT
    return {
        'x': x[np.newaxis, :], 'y': y[np.newaxis, :],
        'vx': vx[np.newaxis, :], 'vy': vy[np.newaxis, :],
        'speed': speed[np.newaxis, :], 'head_direction': hd[np.newaxis, :],
        'd_N': d_N[np.newaxis, :], 'd_S': d_S[np.newaxis, :],
        'd_E': d_E[np.newaxis, :], 'd_W': d_W[np.newaxis, :],
        'd_min': d_min[np.newaxis, :], 't': t[np.newaxis, :],
    }


def _call_generator_single_step(gen, extra_np):
    """Call generator with single-step 2D tensor [batch, 4] (as used in NetworkRNN)."""
    import tensorflow as tf
    # generators expect [batch, 4] for extra_inputs
    extra_tensor = tf.constant(extra_np.astype(np.float32))  # [n, 4]
    t_tensor = tf.constant(np.arange(extra_np.shape[0], dtype=np.float32)[:, None])  # [n, 1]
    return gen(t_tensor, extra_inputs=extra_tensor).numpy()


class TestPrecomputeInputs:
    def test_output_shape(self, sample_trajectory):
        inputs = precompute_inputs(sample_trajectory)
        assert inputs.shape == (1, sample_trajectory['x'].shape[1], config.N_INPUTS)

    def test_all_non_negative(self, sample_trajectory):
        inputs = precompute_inputs(sample_trajectory)
        assert np.all(inputs >= 0)

    def test_d_far_correlation(self, sample_trajectory):
        inputs = precompute_inputs(sample_trajectory)
        d_min = sample_trajectory['d_min'][0]
        assert np.corrcoef(inputs[0, :, 0], d_min)[0, 1] > 0.5

    def test_d_near_correlation(self, sample_trajectory):
        inputs = precompute_inputs(sample_trajectory)
        d_min = sample_trajectory['d_min'][0]
        assert np.corrcoef(inputs[0, :, 1], d_min)[0, 1] < -0.5

    def test_cb_channels_shape(self, sample_trajectory):
        inputs = precompute_inputs(sample_trajectory)
        cb = inputs[0, :, 3:3 + config.N_CB]
        assert cb.shape == (sample_trajectory['x'].shape[1], config.N_CB)
        assert np.all(cb >= 0)
        assert cb.max() <= config.F_MAX_CB + 1e-4

    def test_cdhd_channels_shape(self, sample_trajectory):
        inputs = precompute_inputs(sample_trajectory)
        cdhd = inputs[0, :, 3 + config.N_CB:3 + config.N_CB + config.N_HD]
        assert cdhd.shape == (sample_trajectory['x'].shape[1], config.N_HD)
        assert np.all(cdhd >= 0)

    def test_cdhd_zero_at_center(self, sample_trajectory):
        """CD×HD modulation g(CD) is zero at the arena center."""
        cx, cy = config.ARENA_CM / 2.0, config.ARENA_CM / 2.0
        traj = {
            'x': np.array([[cx]]), 'y': np.array([[cy]]),
            'vx': np.array([[0.0]]), 'vy': np.array([[0.0]]),
            'speed': np.array([[0.0]]), 'head_direction': np.array([[0.0]]),
            'd_N': np.array([[cy]]), 'd_S': np.array([[cy]]),
            'd_E': np.array([[cx]]), 'd_W': np.array([[cx]]),
            'd_min': np.array([[min(cx, cy)]]), 't': np.array([[0.0]]),
        }
        inputs = precompute_inputs(traj)
        cdhd = inputs[0, 0, 3 + config.N_CB:3 + config.N_CB + config.N_HD]
        # At center CD=0 → g(CD)=0 → CD×HD channels all zero
        np.testing.assert_array_almost_equal(cdhd, np.zeros_like(cdhd))

    def test_cd_channels(self, sample_trajectory):
        inputs = precompute_inputs(sample_trajectory)
        cd_far_idx = 3 + config.N_CB + config.N_HD
        cd_near_idx = cd_far_idx + 1
        cd_far = inputs[0, :, cd_far_idx]
        cd_near = inputs[0, :, cd_near_idx]
        # CD computed from x, y
        cx, cy = config.ARENA_CM / 2.0, config.ARENA_CM / 2.0
        x = sample_trajectory['x'][0]
        y = sample_trajectory['y'][0]
        CD = np.sqrt((cx - x)**2 + (cy - y)**2)
        # cd_far should be positively correlated with CD
        assert np.corrcoef(cd_far, CD)[0, 1] > 0.99
        # cd_near should be negatively correlated with CD
        assert np.corrcoef(cd_near, CD)[0, 1] < -0.99

    def test_cb_at_corner_facing_center(self, sample_trajectory):
        """At SW corner facing NE, CB≈0 → CB cell with θ_pref=0 should be max."""
        cx, cy = config.ARENA_CM / 2.0, config.ARENA_CM / 2.0
        heading = np.arctan2(cy, cx)
        traj = {
            'x': np.array([[1e-3]]), 'y': np.array([[1e-3]]),
            'vx': np.array([[np.cos(heading)]]),
            'vy': np.array([[np.sin(heading)]]),
            'speed': np.array([[1.0]]),
            'head_direction': np.array([[heading]]),
            'd_N': np.array([[cy - 1e-3]]), 'd_S': np.array([[1e-3]]),
            'd_E': np.array([[cx - 1e-3]]), 'd_W': np.array([[1e-3]]),
            'd_min': np.array([[1e-3]]), 't': np.array([[0.0]]),
        }
        inputs = precompute_inputs(traj)
        cb = inputs[0, 0, 3:3 + config.N_CB]
        # CB≈0 → cell k with θ_pref_k=0 (deg) should be ≈ F_MAX_CB
        assert cb[0] == cb.max()
        np.testing.assert_allclose(cb[0], config.F_MAX_CB, atol=1e-4)


class TestVisualizationHelpers:
    """Tests for the article-formula helpers in visualize_dataset.py."""

    def test_compute_cb_zero_at_corner_facing_center(self):
        from visualize_dataset import compute_cb
        cx, cy = config.ARENA_CM / 2.0, config.ARENA_CM / 2.0
        traj = {
            'x': np.array([[1e-3]]), 'y': np.array([[1e-3]]),
            'head_direction': np.array([[np.arctan2(cy, cx)]]),
        }
        cb = compute_cb(traj)
        np.testing.assert_allclose(cb, 0.0, atol=1e-6)

    def test_compute_cd_zero_at_center(self):
        from visualize_dataset import compute_cd
        cx, cy = config.ARENA_CM / 2.0, config.ARENA_CM / 2.0
        traj = {
            'x': np.array([[cx]]), 'y': np.array([[cy]]),
        }
        cd = compute_cd(traj)
        np.testing.assert_allclose(cd, 0.0, atol=1e-6)

    def test_circular_mvl_perfect_for_constant_bearing(self):
        from visualize_dataset import circular_mvl
        # 4 timesteps, all at bearing 0, varying per-cell weights
        signals = np.array([[1.0, 0.0, 2.0, 0.5],
                            [0.5, 0.0, 1.0, 0.3],
                            [2.0, 0.0, 3.0, 0.7],
                            [0.5, 0.0, 1.5, 0.5]])
        angles = np.array([0.0, 0.0, 0.0, 0.0])
        mvl, pref = circular_mvl(signals, angles)
        # Cells with non-zero total weight → MVL = 1, pref = 0
        np.testing.assert_allclose(mvl[[0, 2, 3]], 1.0, atol=1e-6)
        # Cell with all-zero weight → 0/0 handled as 0
        np.testing.assert_allclose(mvl[1], 0.0, atol=1e-6)
        np.testing.assert_allclose(pref[[0, 2, 3]], 0.0, atol=1e-6)

    def test_circular_mvl_zero_for_orthogonal_signals(self):
        from visualize_dataset import circular_mvl
        # Half the timesteps at bearing 0, half at π → MVL = 0
        signals = np.array([[1.0], [1.0], [1.0], [1.0]])
        angles = np.array([0.0, 0.0, np.pi, np.pi])
        mvl, pref = circular_mvl(signals, angles)
        np.testing.assert_allclose(mvl, 0.0, atol=1e-6)

    def test_linear_fit_R2_perfect_linear(self):
        from visualize_dataset import linear_fit_R2
        predictor = np.linspace(0, 10, 100)
        signals = np.stack([2.0 * predictor + 1.0,
                            -3.0 * predictor + 5.0], axis=-1)
        R2, slope = linear_fit_R2(signals, predictor)
        np.testing.assert_allclose(R2, [1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(slope, [2.0, -3.0], atol=1e-6)

    def test_compute_spatial_rate_map_shape_and_peak_at_wall(self):
        from visualize_dataset import compute_spatial_rate_map
        n = 50
        # Trajectory that hugs the N wall
        x = np.linspace(1, 49, n)[None, :]
        y = np.full((1, n), config.ARENA_CM - 0.5)
        sig = np.ones((1, n)) * 10.0
        traj = {'x': x, 'y': y}
        rmap = compute_spatial_rate_map(traj, sig[0], bin_size=1.0)
        assert rmap.shape == (50, 50)
        # Peak should be at the top row (y_idx=49, near N wall)
        assert rmap[49].max() > 5.0
        assert rmap[0].max() == 0.0  # nothing at S wall

    def test_compute_border_score_border_cell_high_b(self):
        from visualize_dataset import compute_spatial_rate_map, compute_border_score
        # Trajectory fills a 5-row strip near the N wall (5*50 = 250 cm² > 200)
        n = 250
        x = np.tile(np.linspace(1, 49, 50), 5)
        y = np.repeat(np.linspace(config.ARENA_CM - 5, config.ARENA_CM - 1, 5), 50)
        sig = np.ones(n) * 10.0
        traj = {'x': x[None, :], 'y': y[None, :]}
        rmap = compute_spatial_rate_map(traj, sig, bin_size=1.0)
        bs = compute_border_score(rmap)
        assert bs['b'] > 0.5, f"expected b > 0.5 for wall cell, got {bs['b']}"
        assert bs['b_robust'] > 0.0
        assert bs['n_fields'] >= 1

    def test_compute_border_score_center_cell_negative_b(self):
        from visualize_dataset import compute_spatial_rate_map, compute_border_score
        # Trajectory covers a 15x15 cm region at the center (225 bins > 200)
        n = 225
        xs, ys = np.meshgrid(np.linspace(18, 32, 15), np.linspace(18, 32, 15))
        x = xs.flatten()
        y = ys.flatten()
        x = np.tile(x, n // len(x) + 1)[:n]
        y = np.tile(y, n // len(y) + 1)[:n]
        sig = np.ones(n) * 10.0
        traj = {'x': x[None, :], 'y': y[None, :]}
        rmap = compute_spatial_rate_map(traj, sig, bin_size=1.0)
        bs = compute_border_score(rmap)
        # Center cell: field is far from any wall → d_m >> 0, c_M small → b < 0
        assert bs['b'] < 0.0, f"expected b < 0 for center cell, got {bs['b']}"
        assert bs['b_robust'] < 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
