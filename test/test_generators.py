"""Tests for input generators and precompute_inputs."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.inputs import (
    DistanceFarGenerator, DistanceNearGenerator, SpeedGenerator,
    HeadDirectionGenerator, precompute_inputs,
)


@pytest.fixture
def sample_trajectory():
    """Create a simple synthetic trajectory for testing."""
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
        'x': x, 'y': y, 'vx': vx, 'vy': vy,
        'speed': speed, 'head_direction': hd,
        'd_N': d_N, 'd_S': d_S, 'd_E': d_E, 'd_W': d_W,
        'd_min': d_min, 't': t,
    }


def _call_generator_single_step(gen, extra_np):
    """Call generator with single-step 2D tensor [batch, 4] (as used in NetworkRNN)."""
    import tensorflow as tf
    # generators expect [batch, 4] for extra_inputs
    extra_tensor = tf.constant(extra_np.astype(np.float32))  # [n, 4]
    t_tensor = tf.constant(np.arange(extra_np.shape[0], dtype=np.float32)[:, None])  # [n, 1]
    return gen(t_tensor, extra_inputs=extra_tensor).numpy()


class TestDistanceFarGenerator:
    def test_n_units(self):
        gen = DistanceFarGenerator(name="d_far")
        assert gen.n_units == 1

    def test_non_negative(self, sample_trajectory):
        from utils.dataset import trajectory_to_extra_inputs
        extra = trajectory_to_extra_inputs(sample_trajectory)
        gen = DistanceFarGenerator(name="d_far")
        rates = _call_generator_single_step(gen, extra)
        assert np.all(rates >= 0)

    def test_monotonic_increasing(self, sample_trajectory):
        from utils.dataset import trajectory_to_extra_inputs
        extra = trajectory_to_extra_inputs(sample_trajectory)
        gen = DistanceFarGenerator(name="d_far")
        rates = _call_generator_single_step(gen, extra).flatten()
        d_min = sample_trajectory['d_min']
        assert np.corrcoef(rates, d_min)[0, 1] > 0.5


class TestDistanceNearGenerator:
    def test_non_negative(self, sample_trajectory):
        from utils.dataset import trajectory_to_extra_inputs
        extra = trajectory_to_extra_inputs(sample_trajectory)
        gen = DistanceNearGenerator(name="d_near")
        rates = _call_generator_single_step(gen, extra)
        assert np.all(rates >= 0)

    def test_monotonic_decreasing(self, sample_trajectory):
        from utils.dataset import trajectory_to_extra_inputs
        extra = trajectory_to_extra_inputs(sample_trajectory)
        gen = DistanceNearGenerator(name="d_near")
        rates = _call_generator_single_step(gen, extra).flatten()
        d_min = sample_trajectory['d_min']
        assert np.corrcoef(rates, d_min)[0, 1] < -0.5


class TestSpeedGenerator:
    def test_baseline(self, sample_trajectory):
        from utils.dataset import trajectory_to_extra_inputs
        extra = trajectory_to_extra_inputs(sample_trajectory)
        gen = SpeedGenerator(name="v")
        rates = _call_generator_single_step(gen, extra)
        assert np.all(rates >= config.BETA_0)


class TestHeadDirectionGenerator:
    def test_n_units(self):
        gen = HeadDirectionGenerator(name="hd")
        assert gen.n_units == config.N_HD

    def test_output_shape(self, sample_trajectory):
        from utils.dataset import trajectory_to_extra_inputs
        extra = trajectory_to_extra_inputs(sample_trajectory)
        gen = HeadDirectionGenerator(name="hd")
        rates = _call_generator_single_step(gen, extra)
        assert rates.shape == (len(sample_trajectory['x']), config.N_HD)

    def test_non_negative(self, sample_trajectory):
        from utils.dataset import trajectory_to_extra_inputs
        extra = trajectory_to_extra_inputs(sample_trajectory)
        gen = HeadDirectionGenerator(name="hd")
        rates = _call_generator_single_step(gen, extra)
        assert np.all(rates >= 0)


class TestPrecomputeInputs:
    def test_output_shape(self, sample_trajectory):
        inputs = precompute_inputs(sample_trajectory)
        assert inputs.shape == (len(sample_trajectory['x']), config.N_INPUTS)

    def test_all_non_negative(self, sample_trajectory):
        inputs = precompute_inputs(sample_trajectory)
        assert np.all(inputs >= 0)

    def test_d_far_correlation(self, sample_trajectory):
        inputs = precompute_inputs(sample_trajectory)
        d_min = sample_trajectory['d_min']
        assert np.corrcoef(inputs[:, 0], d_min)[0, 1] > 0.5

    def test_d_near_correlation(self, sample_trajectory):
        inputs = precompute_inputs(sample_trajectory)
        d_min = sample_trajectory['d_min']
        assert np.corrcoef(inputs[:, 1], d_min)[0, 1] < -0.5

    def test_hd_channels(self, sample_trajectory):
        inputs = precompute_inputs(sample_trajectory)
        hd = inputs[:, 3:21]
        assert hd.shape[1] == 18
        assert np.all(hd >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
