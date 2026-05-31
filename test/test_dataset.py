"""Tests for dataset preparation, batching, and HDF5 save/load."""

import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.dataset import (
    trajectory_to_extra_inputs, compute_targets, make_t_sequence,
    prepare_batches, save_dataset_hdf5, load_dataset_hdf5,
)


@pytest.fixture
def sample_data():
    """Create sample inputs and targets."""
    n_steps = 10000
    rng = np.random.RandomState(42)
    inputs = rng.uniform(0, 15, (n_steps, config.N_INPUTS)).astype(np.float32)
    targets = rng.uniform(0, config.F_MAX_BORDER, (n_steps, 4)).astype(np.float32)
    return inputs, targets


@pytest.fixture
def sample_trajectory():
    """Create a simple synthetic trajectory."""
    n = 10000
    rng = np.random.RandomState(42)
    x = rng.uniform(5, config.ARENA_CM - 5, n)
    y = rng.uniform(5, config.ARENA_CM - 5, n)
    speed = rng.uniform(5, 40, n)
    hd = rng.uniform(-np.pi, np.pi, n)
    d_N = config.ARENA_CM - y
    d_S = y
    d_E = config.ARENA_CM - x
    d_W = x
    return {
        'x': x, 'y': y, 'speed': speed, 'head_direction': hd,
        'vx': speed * np.cos(hd), 'vy': speed * np.sin(hd),
        'd_N': d_N, 'd_S': d_S, 'd_E': d_E, 'd_W': d_W,
        'd_min': np.minimum(np.minimum(d_N, d_S), np.minimum(d_E, d_W)),
        't': np.arange(n) * config.TRAJECTORY_DT,
    }


class TestTrajectoryToExtraInputs:
    def test_shape(self, sample_trajectory):
        extra = trajectory_to_extra_inputs(sample_trajectory)
        assert extra.shape == (len(sample_trajectory['x']), 4)

    def test_xy_match(self, sample_trajectory):
        extra = trajectory_to_extra_inputs(sample_trajectory)
        np.testing.assert_array_almost_equal(extra[:, 0], sample_trajectory['x'])
        np.testing.assert_array_almost_equal(extra[:, 1], sample_trajectory['y'])


class TestComputeTargets:
    def test_shape(self, sample_trajectory):
        targets = compute_targets(sample_trajectory)
        assert targets.shape == (len(sample_trajectory['x']), 4)

    def test_range(self, sample_trajectory):
        targets = compute_targets(sample_trajectory)
        assert np.all(targets >= 0)
        assert np.all(targets <= config.F_MAX_BORDER + 0.01)

    def test_max_at_wall(self):
        """Target should be max when agent is at wall (d=0)."""
        traj = {
            'x': np.array([0.0]),
            'y': np.array([0.0]),
            'd_N': np.array([config.ARENA_CM]),
            'd_S': np.array([0.0]),
            'd_E': np.array([config.ARENA_CM]),
            'd_W': np.array([0.0]),
        }
        targets = compute_targets(traj)
        # S and W walls should be at max
        assert targets[0, 1] > targets[0, 0]  # S > N
        assert targets[0, 3] > targets[0, 2]  # W > E


class TestMakeTSequence:
    def test_shape(self):
        t = make_t_sequence(100)
        assert t.shape == (1, 100, 1)

    def test_values(self):
        t = make_t_sequence(10)
        expected = np.arange(10, dtype=np.float32) * config.SIM_DT
        np.testing.assert_array_almost_equal(t[0, :, 0], expected)


class TestPrepareBatches:
    def test_batch_count(self, sample_data):
        inputs, targets = sample_data
        batch_steps = int(config.BATCH_DURATION / (config.DT / 1000))
        expected_batches = len(inputs) // batch_steps
        batches = prepare_batches(inputs, targets)
        assert len(batches) == expected_batches

    def test_batch_shapes(self, sample_data):
        inputs, targets = sample_data
        batches = prepare_batches(inputs, targets)
        batch_steps = int(config.BATCH_DURATION / (config.DT / 1000))
        for b in batches:
            assert b['t_seq'].shape == (1, batch_steps, 1)
            assert b['inputs'].shape == (1, batch_steps, config.N_INPUTS)
            assert b['targets'].shape == (1, batch_steps, 4)

    def test_total_steps_preserved(self, sample_data):
        inputs, targets = sample_data
        batches = prepare_batches(inputs, targets)
        total = sum(b['inputs'].shape[1] for b in batches)
        batch_steps = int(config.BATCH_DURATION / (config.DT / 1000))
        assert total == len(batches) * batch_steps

    def test_sequential_data(self, sample_data):
        inputs, targets = sample_data
        batches = prepare_batches(inputs, targets)
        # First batch inputs should match first slice
        np.testing.assert_array_almost_equal(
            batches[0]['inputs'][0], inputs[:batches[0]['inputs'].shape[1]])


class TestHDF5SaveLoad:
    def test_roundtrip(self, sample_data):
        inputs, targets = sample_data
        batches = prepare_batches(inputs, targets)

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name
        try:
            save_dataset_hdf5(path, batches, {'test_key': 42})
            ds = load_dataset_hdf5(path)

            assert ds['n_batches'] == len(batches)
            assert ds['metadata']['test_key'] == 42

            loaded = ds['get_batch'](0)
            np.testing.assert_array_almost_equal(
                loaded['t_seq'], batches[0]['t_seq'])
            np.testing.assert_array_almost_equal(
                loaded['inputs'], batches[0]['inputs'])
            np.testing.assert_array_almost_equal(
                loaded['targets'], batches[0]['targets'])

            ds['file'].close()
        finally:
            os.unlink(path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
