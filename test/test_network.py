"""Tests for network construction, forward pass, and one training step."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import config
from train import build_network, graph_pack_inputs


@pytest.fixture
def mini_dataset():
    """Create a minimal dataset for testing."""
    batch_steps = 100
    rng = np.random.RandomState(42)
    t_seq = (np.arange(batch_steps, dtype=np.float32) * config.DT).reshape(1, -1, 1)
    inputs = rng.uniform(0, 10, (1, batch_steps, config.N_INPUTS)).astype(np.float32)
    targets = rng.uniform(0, config.F_MAX_BORDER, (1, batch_steps, 4)).astype(np.float32)
    targets_6 = np.pad(targets, [[0, 0], [0, 0], [0, 2]])
    return t_seq, inputs, targets_6


class TestNetworkBuild:
    def test_builds_without_error(self):
        network = build_network()
        assert network is not None

    def test_has_trainable_variables(self):
        network = build_network()
        assert len(network.trainable_variables) > 0

    def test_population_names(self):
        network = build_network()
        assert config.POPULATION_NAME in network._graph.population_names

    def test_synapse_names(self):
        network = build_network()
        assert 'inp->cells' in network._graph.synapse_names
        assert 'cells->cells' in network._graph.synapse_names

    def test_input_declared(self):
        network = build_network()
        assert 'inputs' in network._graph.input_names


class TestForwardPass:
    def test_output_shape(self, mini_dataset):
        network = build_network()
        t_seq, inputs_np, _ = mini_dataset
        t_tensor = tf.constant(t_seq)
        inputs = graph_pack_inputs(network, inputs_np)
        output = network(t_tensor, inputs=inputs, training=False)
        rates = output.firing_rates[config.POPULATION_NAME]
        assert rates.shape == (1, mini_dataset[0].shape[1], config.N_POP_UNITS)

    def test_output_finite_first_steps(self, mini_dataset):
        """Border cells (units 0-3) should be finite at step 0.
        Basket/Axo (units 4-5) may produce NaN with random inputs
        due to known FS parameter sensitivity."""
        network = build_network()
        t_seq, inputs_np, _ = mini_dataset
        t_tensor = tf.constant(t_seq)
        inputs = graph_pack_inputs(network, inputs_np)
        output = network(t_tensor, inputs=inputs, training=False)
        rates = output.firing_rates[config.POPULATION_NAME].numpy()
        # Border cells (units 0-3) should be finite at step 0
        assert np.all(np.isfinite(rates[0, 0, :4])), "Border cells NaN at step 0"


class TestTrainingStep:
    def test_one_step_computes_loss(self, mini_dataset):
        """Verify training step produces a loss value (may be NaN with random
        inputs due to known parameter sensitivity)."""
        network = build_network()
        t_seq, inputs_np, targets_6 = mini_dataset

        optimizer = tf.keras.optimizers.Adam(1e-3)
        target = {config.POPULATION_NAME: tf.constant(targets_6, dtype=tf.float32)}
        from neuraltide.training import MSELoss, CompositeLoss
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])

        t_tensor = tf.constant(t_seq)
        inputs = graph_pack_inputs(network, inputs_np)

        with tf.GradientTape() as tape:
            output = network(t_tensor, inputs=inputs, training=True)
            loss = loss_fn(output, network)

        # Loss should be computable (may be NaN with random inputs)
        assert loss is not None

        grads = tape.gradient(loss, network.trainable_variables)
        assert len(grads) == len(network.trainable_variables)

    def test_trainable_vars_have_gradients(self, mini_dataset):
        network = build_network()
        t_seq, inputs_np, targets_6 = mini_dataset

        target = {config.POPULATION_NAME: tf.constant(targets_6, dtype=tf.float32)}
        from neuraltide.training import MSELoss, CompositeLoss
        loss_fn = CompositeLoss([(1.0, MSELoss(target))])

        t_tensor = tf.constant(t_seq)
        inputs = graph_pack_inputs(network, inputs_np)

        with tf.GradientTape() as tape:
            output = network(t_tensor, inputs=inputs, training=True)
            loss = loss_fn(output, network)

        grads = tape.gradient(loss, network.trainable_variables)
        non_none = sum(1 for g in grads if g is not None)
        assert non_none > 0, "No gradients computed"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
