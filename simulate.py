"""Border cell simulation: network construction and training."""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Callable

import neuraltide as nt
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.integrators import EulerIntegrator, HeunIntegrator, RK4Integrator
import h5py

import config
from utils.csv_loader import (
    get_synapse_params_for_connection,
    get_izhikevich_dimensionless_params,
    get_neuron_ei,
    get_neuron_vr,
)
from utils.inputs import DistanceFarGenerator, DistanceNearGenerator, SpeedGenerator, HeadDirectionGenerator
from utils.dataset import prepare_batch
from utils.trajectory import TrajectoryGenerator

tf.random.set_seed(config.RANDOM_SEED)
nt.seed_everything(config.RANDOM_SEED)

# ============================================================
# Population parameters (per-unit, from CSV)
# ============================================================

def _make_merged_params() -> dict:
    """Build per-unit params dict for single n_units=6 population.

    Each param is a list of 6 values, one per unit.
    """
    per_unit = {k: [] for k in [
        'tau_pop', 'alpha', 'a', 'b', 'w_jump', 'Delta_I', 'I_ext']}
    for name in config.UNIT_NAMES:
        ntype = config.UNIT_TYPE[name]
        p = get_izhikevich_dimensionless_params(
            config.NEURON_TYPE_MAP[ntype])
        for k in per_unit:
            per_unit[k].append(p[k])
    return per_unit


def _make_pconn_mask(n_pre: int, n_post: int,
                     src_idx: int, tgt_idx: int) -> np.ndarray:
    """Create a pconn matrix with a single connection at (src_idx, tgt_idx)."""
    m = np.zeros((n_pre, n_post), dtype=np.float64)
    m[src_idx, tgt_idx] = 1.0
    return m


# ============================================================
# Synapse parameter helpers
# ============================================================

def _make_ts_params_dict(conn_key: str, n_pre: int, n_post: int) -> dict:
    """Build TsodyksMarkram params with trainable flags and e_r."""
    try:
        csv_params = get_synapse_params_for_connection(conn_key)
        g = csv_params['gsyn_max']
        tf_val = csv_params['tau_f']
        td = csv_params['tau_d']
        tr = csv_params['tau_r']
        u = csv_params['Uinc']
    except (ValueError, KeyError):
        if 'Pyramidal' in conn_key.split('→')[0]:
            d = config.TM_SYN_DEFAULTS['Exc→Exc']
        else:
            d = config.TM_SYN_DEFAULTS['Inh→Exc']
        g, td, tr, tf_val, u = d['gsyn_max'], d['tau_d'], d['tau_r'], d['tau_f'], d['Uinc']

    g_scaled = g * config.GSYN_SCALE_DIMENSIONAL
    base_g_perturbed = g_scaled * (1.0 + tf.random.uniform([], -0.3, 0.3).numpy())

    pre_short, post_short = conn_key.split('→')
    if pre_short == 'Input':
        csv_tuple = config.SYNAPSE_TYPE_MAP.get(conn_key)
        pre_csv = csv_tuple[1] if csv_tuple else "CA1 Back-Projection"
    else:
        pre_csv = config.NEURON_TYPE_MAP[pre_short]
    post_csv = config.NEURON_TYPE_MAP[post_short]

    pre_ei = get_neuron_ei(pre_csv)
    E_r = 0.0 if pre_ei == 'e' else -75.0
    Vr_post = get_neuron_vr(post_csv)
    e_r = 1.0 + E_r / abs(Vr_post)

    return {
        'gsyn_max': {
            'value': max(0.001, base_g_perturbed),
            'trainable': config.TRAIN_SYNAPSE_GMAX,
            'min': 0.0,
        },
        'tau_f': {
            'value': float(tf_val),
            'trainable': config.TRAIN_SYNAPSE_TAU,
            'min': 1.0,
        },
        'tau_d': {
            'value': float(td),
            'trainable': config.TRAIN_SYNAPSE_TAU,
            'min': 1.0,
        },
        'tau_r': {
            'value': float(tr),
            'trainable': config.TRAIN_SYNAPSE_TAU,
            'min': 1.0,
        },
        'Uinc': {
            'value': float(u),
            'trainable': config.TRAIN_SYNAPSE_U,
            'min': 0.0, 'max': 1.0,
        },
        'pconn': {
            'value': 1.0,
            'trainable': False,
        },
        'e_r': {
            'value': e_r,
            'trainable': False,
        },
    }


# ============================================================
# Network construction
# ============================================================

def build_network() -> NetworkGraph:
    """Build the border cell network graph with vectorized single population."""
    dt = config.SIM_DT
    graph = NetworkGraph(dt=dt)

    # — Single dynamic population (n_units=6) —
    merged_params = _make_merged_params()
    pop = IzhikevichMeanField(
        dt=dt, params=merged_params, name=config.POPULATION_NAME)
    graph.add_population(name=config.POPULATION_NAME, model=pop)

    # — Input populations —
    graph.add_input_population(
        name="d_far", generator=DistanceFarGenerator(name="d_far"))
    graph.add_input_population(
        name="d_near", generator=DistanceNearGenerator(name="d_near"))
    graph.add_input_population(
        name="v", generator=SpeedGenerator(name="v"))
    graph.add_input_population(
        name="hd", generator=HeadDirectionGenerator(name="hd"))

    input_names = ['d_far', 'd_near', 'v', 'hd']

    # — Recurrent synapses (all-to-all among 6 units) —
    for src_name in config.UNIT_NAMES:
        for tgt_name in config.UNIT_NAMES:
            src_idx = config.UNIT_IDX[src_name]
            tgt_idx = config.UNIT_IDX[tgt_name]
            pre_type = config.UNIT_TYPE[src_name]
            post_type = config.UNIT_TYPE[tgt_name]
            conn_key = f'{pre_type}→{post_type}'
            ts_params = _make_ts_params_dict(conn_key, config.N_UNITS, config.N_UNITS)
            ts_params['pconn'] = {
                'value': _make_pconn_mask(config.N_UNITS, config.N_UNITS, src_idx, tgt_idx),
                'trainable': False,
            }
            syn = TsodyksMarkramSynapse(
                n_pre=config.N_UNITS, n_post=config.N_UNITS, dt=dt,
                params=ts_params,
                name=f'syn_rec_{src_name}_to_{tgt_name}',
            )
            graph.add_synapse(
                name=f'syn_rec_{src_name}_to_{tgt_name}',
                model=syn, src=config.POPULATION_NAME, tgt=config.POPULATION_NAME)

    # — Input-to-population synapses —
    for inp_name in input_names:
        n_pre = config.N_HD if inp_name == 'hd' else 1
        for tgt_name in config.UNIT_NAMES:
            tgt_idx = config.UNIT_IDX[tgt_name]
            post_type = config.UNIT_TYPE[tgt_name]
            conn_key = f'Input→{post_type}'
            ts_params = _make_ts_params_dict(conn_key, n_pre, config.N_UNITS)
            ts_params['pconn'] = {
                'value': _make_pconn_mask(n_pre, config.N_UNITS, 0, tgt_idx),
                'trainable': False,
            }
            syn = TsodyksMarkramSynapse(
                n_pre=n_pre, n_post=config.N_UNITS, dt=dt,
                params=ts_params,
                name=f'syn_inp_{inp_name}_to_{tgt_name}',
            )
            graph.add_synapse(
                name=f'syn_inp_{inp_name}_to_{tgt_name}',
                model=syn, src=inp_name, tgt=config.POPULATION_NAME)

    graph.validate()
    return graph


def build_integrator():
    if config.INTEGRATOR == 'euler':
        return EulerIntegrator()
    elif config.INTEGRATOR == 'heun':
        return HeunIntegrator()
    elif config.INTEGRATOR == 'rk4':
        return RK4Integrator()
    else:
        raise ValueError(f"Unknown integrator: {config.INTEGRATOR}")


# ============================================================
# Loss function
# ============================================================

def compute_loss(firing_rates: Dict[str, tf.Tensor],
                 targets: tf.Tensor) -> tf.Tensor:
    """Compute composite loss for border cell optimization.

    Args:
        firing_rates: dict with single key 'border_cells' [batch, n_steps, 6]
                      units: [Border_N, Border_S, Border_E, Border_W, Basket, Axo]
        targets: target rates [batch, n_steps, 4] for [N, S, E, W]

    Returns:
        total_loss: scalar
    """
    all_rates = firing_rates.get(config.POPULATION_NAME)
    if all_rates is None:
        all_rates = tf.zeros_like(targets[..., 0:1])

    # Border cells are units 0-3
    preds = all_rates[..., :4]  # [batch, n_steps, 4]

    # MSE loss per wall
    mse = tf.reduce_mean(tf.square(preds - targets))

    # Firing rate regularization
    mean_pred = tf.reduce_mean(preds, axis=[0, 1])
    mean_target = tf.reduce_mean(targets, axis=[0, 1])
    l_fr = tf.reduce_mean(tf.abs(mean_pred - mean_target))

    # Sparsity loss (lifetime sparseness)
    mean_sq = tf.square(tf.reduce_mean(preds, axis=1))
    sq_mean = tf.reduce_mean(tf.square(preds), axis=1)
    sparsity = 1.0 - tf.reduce_mean(mean_sq / (sq_mean + 1e-8))
    l_sp = tf.reduce_mean(sparsity)

    total = (config.LOSS_WEIGHT_MSE * mse +
             config.LOSS_WEIGHT_FR * l_fr +
             config.LOSS_WEIGHT_SPARSITY * l_sp)

    return total, {'mse': mse, 'l_fr': l_fr, 'l_sp': l_sp}


# ============================================================
# Training
# ============================================================

class SimulationRunner:
    """Manages network simulation and training."""

    def __init__(self):
        self.graph = build_network()
        self.integrator = build_integrator()
        self.net_rnn = NetworkRNN(
            self.graph,
            self.integrator,
            return_hidden_states=False,
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE
        )

        self.loss_history = []
        self.mse_history = []
        self.fr_history = []
        self.sp_history = []

        os.makedirs(config.RESULTS_DIR, exist_ok=True)

    @staticmethod
    def precompute_batches(n_batches: int, trial_duration: float, path: str):
        """Load full trajectory once, split into n_batches sequential slices, save to HDF5."""
        use_hdf5 = os.path.exists(config.TRAJECTORY_HDF5)
        gen = None if use_hdf5 else TrajectoryGenerator(seed=config.RANDOM_SEED)

        # compute total steps and steps per batch from trajectory
        if use_hdf5:
            with h5py.File(config.TRAJECTORY_HDF5, 'r') as f:
                total_steps = len(f['x'])
        else:
            raw = generate_trajectory_batch(gen, trial_duration, n_trajectories=1)
            single = {k: raw[k][0] for k in raw}
            interped = interpolate_trajectory(single, config.DT / 1000.0)
            total_steps = len(interped['x'])

        steps_per_batch = total_steps // n_batches
        print(f"Precomputing {n_batches} batches: {steps_per_batch} steps each "
              f"(total {total_steps} steps)...")

        with h5py.File(path, 'w') as f:
            for i in range(n_batches):
                start = i * steps_per_batch
                batch = prepare_batch(gen, trial_duration, config.BATCH_SIZE,
                                      start_step=start, n_steps=steps_per_batch)
                grp = f.create_group(f'batch_{i}')
                grp.create_dataset('t_seq', data=batch['t_seq'])
                grp.create_dataset('extra_seq', data=batch['extra_seq'])
                grp.create_dataset('targets', data=batch['targets'])
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{n_batches} batches saved")
        print(f"  All {n_batches} batches → {path}")

    @staticmethod
    def load_batch(hdf5_path: str, index: int) -> dict:
        """Load a single batch from HDF5 by index."""
        with h5py.File(hdf5_path, 'r') as f:
            grp = f[f'batch_{index}']
            return {
                't_seq': grp['t_seq'][:],
                'extra_seq': grp['extra_seq'][:],
                'targets': grp['targets'][:],
            }

    def train_step(self, t_seq: np.ndarray, extra_seq: np.ndarray,
                   targets: np.ndarray) -> dict:
        """Single training step with gradient descent."""
        t_tensor = tf.constant(t_seq, dtype=tf.float32)
        extra_tensor = tf.constant(extra_seq, dtype=tf.float32)
        target_tensor = tf.constant(targets, dtype=tf.float32)

        if config.GRAD_METHOD == 'adjoint':
            return self._train_step_adjoint(t_tensor, extra_tensor, target_tensor)
        else:
            return self._train_step_bptt(t_tensor, extra_tensor, target_tensor)

    def _train_step_bptt(self, t_seq, extra_seq, targets):
        with tf.GradientTape() as tape:
            output = self.net_rnn(t_seq, extra_inputs_seq=extra_seq, training=True)
            loss, components = compute_loss(output.firing_rates, targets)

        grads = tape.gradient(loss, self.net_rnn.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.net_rnn.trainable_variables)
        )
        return {'loss': loss.numpy(), 'mse': components['mse'].numpy(),
                'l_fr': components['l_fr'].numpy(), 'l_sp': components['l_sp'].numpy()}

    def _train_step_adjoint(self, t_seq, extra_seq, targets):
        """Training step using adjoint state method.

        Simulates forward, then computes adjoint gradients.
        """
        with tf.GradientTape(persistent=False) as tape:
            output = self.net_rnn(t_seq, extra_inputs_seq=extra_seq, training=True)
            loss, components = compute_loss(output.firing_rates, targets)

        grads = tape.gradient(loss, self.net_rnn.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.net_rnn.trainable_variables)
        )
        return {'loss': loss.numpy(), 'mse': components['mse'].numpy(),
                'l_fr': components['l_fr'].numpy(), 'l_sp': components['l_sp'].numpy()}

    def train(self, n_batches: int = None, trial_duration: float = None,
              batches_path: str = None):
        """Run full training loop."""
        n_batches = n_batches or config.N_BATCHES
        trial_duration = trial_duration or config.TRIAL_DURATION
        batches_path = batches_path or os.path.join(config.RESULTS_DIR, 'training_batches.h5')

        if not os.path.exists(batches_path):
            self.precompute_batches(n_batches, trial_duration, batches_path)

        # for trial in range(n_batches):
        #     batch = self.load_batch(batches_path, trial)
        #     metrics = self.train_step(
        #         batch['t_seq'], batch['extra_seq'], batch['targets']
        #     )
        #
        #     self.loss_history.append(float(metrics['loss']))
        #     self.mse_history.append(float(metrics['mse']))
        #     self.fr_history.append(float(metrics['l_fr']))
        #     self.sp_history.append(float(metrics['l_sp']))
        #
        #     if (trial + 1) % config.PRINT_EVERY_N_TRIALS == 0:
        #         print(f"Trial {trial+1:4d}/{n_batches} | "
        #               f"loss={metrics['loss']:.4f} | "
        #               f"mse={metrics['mse']:.4f} | "
        #               f"fr={metrics['l_fr']:.4f} | "
        #               f"sp={metrics['l_sp']:.4f}")
        #
        #     if (trial + 1) % config.SAVE_EVERY_N_TRIALS == 0:
        #         self.save_checkpoint(trial + 1)
        #
        # print("Training complete.")
        # self.save_results()
        return self.loss_history

    def simulate(self, t_seq: np.ndarray, extra_seq: np.ndarray) -> dict:
        """Run forward simulation without training."""
        t_tensor = tf.constant(t_seq, dtype=tf.float32)
        extra_tensor = tf.constant(extra_seq, dtype=tf.float32)
        output = self.net_rnn(t_tensor, extra_inputs_seq=extra_tensor, training=False)
        result = {}
        for name in output.firing_rates:
            result[name] = output.firing_rates[name].numpy()
        return result

    def save_checkpoint(self, trial: int):
        """Save intermediate results to HDF5."""
        path = os.path.join(config.RESULTS_DIR, f'checkpoint_trial_{trial}.h5')
        with h5py.File(path, 'w') as f:
            f.create_dataset('loss_history', data=np.array(self.loss_history))
            f.create_dataset('mse_history', data=np.array(self.mse_history))
            f.create_dataset('fr_history', data=np.array(self.fr_history))
            f.create_dataset('sp_history', data=np.array(self.sp_history))

    def save_results(self):
        """Save final results to HDF5."""
        path = os.path.join(config.RESULTS_DIR, 'results.h5')
        with h5py.File(path, 'w') as f:
            f.create_dataset('loss_history', data=np.array(self.loss_history))
            f.create_dataset('mse_history', data=np.array(self.mse_history))
            f.create_dataset('fr_history', data=np.array(self.fr_history))
            f.create_dataset('sp_history', data=np.array(self.sp_history))

        param_path = os.path.join(config.RESULTS_DIR, 'trained_params.h5')
        with h5py.File(param_path, 'w') as f:
            for v in self.net_rnn.trainable_variables:
                f.create_dataset(v.name.replace(':', '_').replace('/', '_'),
                                 data=v.numpy())

        print(f"Results saved to {config.RESULTS_DIR}/")


def main():
    use_hdf5 = os.path.exists(config.TRAJECTORY_HDF5)
    print(f"Building border cell network ({config.GRAD_METHOD} mode)...")
    if use_hdf5:
        print(f"  Using trajectory from {config.TRAJECTORY_HDF5}")
    runner = SimulationRunner()

    print(f"Network built: 1 population ({config.POPULATION_NAME}, n_units={config.N_UNITS}), "
          f"{len(runner.graph.synapse_names)} synapses")
    print(f"Trainable parameters: "
          f"{np.sum([np.prod(v.shape) for v in runner.net_rnn.trainable_variables])}")
    print(f"Training: {config.N_BATCHES} batches × {config.TRIAL_DURATION}s trajectory...")
    runner.train()

    print("Running final simulation...")
    batches_path = os.path.join(config.RESULTS_DIR, 'training_batches.h5')
    batch = runner.load_batch(batches_path, 0)
    rates = runner.simulate(batch['t_seq'], batch['extra_seq'])

    with h5py.File(os.path.join(config.RESULTS_DIR, 'final_simulation.h5'), 'w') as f:
        for key, value in rates.items():
            f.create_dataset(key, data=value)
        f.create_dataset('t_seq', data=batch['t_seq'])
        f.create_dataset('targets', data=batch['targets'])
    print("Done.")


if __name__ == '__main__':
    main()
