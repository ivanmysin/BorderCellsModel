"""Border cell simulation: network construction and training."""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional

import neuraltide as nt
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.integrators import EulerIntegrator, HeunIntegrator, RK4Integrator

import config
from utils.csv_loader import get_synapse_params_for_connection
from utils.inputs import DistanceFarGenerator, DistanceNearGenerator, SpeedGenerator, HDPopVecGenerator
from utils.dataset import prepare_batch
from utils.trajectory import TrajectoryGenerator

tf.random.set_seed(config.RANDOM_SEED)
nt.seed_everything(config.RANDOM_SEED)

# ============================================================
# Population parameters (MPR dimensionless)
# ============================================================

# RS (Regular Spiking) — Border cell populations
RS_PARAMS = {
    'tau_pop': {'value': 50.0, 'trainable': False},
    'alpha':   {'value': 0.5, 'trainable': False},
    'a':       {'value': 0.02, 'trainable': False},
    'b':       {'value': 0.2, 'trainable': False},
    'w_jump':  {'value': 0.02, 'trainable': False},
    'Delta_I': {'value': 0.5, 'trainable': config.TRAIN_POP_DELTA_I,
                'min': 0.05, 'max': 2.0},
    'I_ext':   {'value': 0.2, 'trainable': config.TRAIN_POP_IEXT,
                'min': 0.01, 'max': 3.0},
}

# RS Border with per-population I_ext perturbation to break symmetry
def _rs_params_for_unit(unit_idx: int) -> dict:
    """RS params with slight I_ext perturbation per unit."""
    import numpy as np
    rng = np.random.default_rng(unit_idx + 42)
    iext = 0.2 + rng.normal(0, 0.05)
    p = dict(RS_PARAMS)
    p['I_ext'] = {'value': max(0.01, iext), 'trainable': config.TRAIN_POP_IEXT,
                  'min': 0.01, 'max': 3.0}
    return p

# FS (Fast Spiking) — Basket and Axo-axonic
FS_PARAMS = {
    'tau_pop': {'value': 10.0, 'trainable': False},
    'alpha':   {'value': 0.5, 'trainable': False},
    'a':       {'value': 0.1, 'trainable': False},
    'b':       {'value': 0.2, 'trainable': False},
    'w_jump':  {'value': 0.01, 'trainable': False},
    'Delta_I': {'value': 0.5, 'trainable': config.TRAIN_POP_DELTA_I,
                'min': 0.05, 'max': 2.0},
    'I_ext':   {'value': 0.5, 'trainable': config.TRAIN_POP_IEXT,
                'min': 0.1, 'max': 5.0},
}

# ============================================================
# Synapse parameter helpers
# ============================================================

def _make_ts_params(conn_key: str, n_pre: int, n_post: int) -> dict:
    """Build TsodyksMarkram parameter dict from CSV or defaults."""
    try:
        csv_params = get_synapse_params_for_connection(conn_key)
        g = csv_params['gsyn_max']
        tf_val = csv_params['tau_f']
        td = csv_params['tau_d']
        tr = csv_params['tau_r']
        u = csv_params['Uinc']
    except (ValueError, KeyError):
        # Use Exc→Exc defaults for Pyramidal source, Inh→* for others
        if 'Pyramidal' in conn_key.split('→')[0]:
            d = config.TM_SYN_DEFAULTS['Exc→Exc']
        else:
            d = config.TM_SYN_DEFAULTS['Inh→Exc']
        g, td, tr, tf_val, u = d['gsyn_max'], d['tau_d'], d['tau_r'], d['tau_f'], d['Uinc']

    # Scale g for dimensionless MPR
    # CSV values are in nS, need dimensionless scale
    g_scaled = g * 0.5

    # Determine e_r based on source type
    if 'Pyramidal' in conn_key.split('→')[0]:
        e_r = 1.0
    else:
        e_r = -0.1

    return {
        'gsyn_max': g_scaled,
        'tau_f': float(tf_val),
        'tau_d': float(td),
        'tau_r': float(tr),
        'Uinc': float(u),
        'pconn': 1.0,
        'e_r': e_r,
    }


def _make_ts_params_dict(conn_key: str, n_pre: int, n_post: int) -> dict:
    """Build TsodyksMarkram params with trainable flags."""
    p = _make_ts_params(conn_key, n_pre, n_post)
    # Add small random perturbation to break symmetry between populations
    base_g = p['gsyn_max']
    g_perturbed = base_g * (1.0 + tf.random.uniform([], -0.3, 0.3).numpy())
    return {
        'gsyn_max': {
            'value': max(0.001, g_perturbed),
            'trainable': config.TRAIN_SYNAPSE_GMAX,
            'min': 0.0,
        },
        'tau_f': {
            'value': p['tau_f'],
            'trainable': config.TRAIN_SYNAPSE_TAU,
            'min': 1.0,
        },
        'tau_d': {
            'value': p['tau_d'],
            'trainable': config.TRAIN_SYNAPSE_TAU,
            'min': 1.0,
        },
        'tau_r': {
            'value': p['tau_r'],
            'trainable': config.TRAIN_SYNAPSE_TAU,
            'min': 1.0,
        },
        'Uinc': {
            'value': p['Uinc'],
            'trainable': config.TRAIN_SYNAPSE_U,
            'min': 0.0, 'max': 1.0,
        },
        'pconn': {
            'value': p['pconn'],
            'trainable': False,
        },
        'e_r': {
            'value': p['e_r'],
            'trainable': False,
        },
    }


# ============================================================
# Network construction
# ============================================================

def build_network() -> NetworkGraph:
    """Build the border cell network graph."""
    dt = config.SIM_DT
    graph = NetworkGraph(dt=dt)

    pop_names = ['Border_N', 'Border_S', 'Border_E', 'Border_W',
                 'Basket', 'Axo']

    # Dynamic populations
    for i, name in enumerate(pop_names):
        if i < 4:
            # Border populations: each gets slightly perturbed params
            params = _rs_params_for_unit(i)
        else:
            params = FS_PARAMS
        pop = IzhikevichMeanField(dt=dt, params=params, name=name)
        graph.add_population(name=name, model=pop)

    # Input populations
    graph.add_input_population(
        name="d_far", generator=DistanceFarGenerator(name="d_far")
    )
    graph.add_input_population(
        name="d_near", generator=DistanceNearGenerator(name="d_near")
    )
    graph.add_input_population(
        name="v", generator=SpeedGenerator(name="v")
    )
    graph.add_input_population(
        name="hd", generator=HDPopVecGenerator(name="hd")
    )

    input_names = ['d_far', 'd_near', 'v', 'hd']

    # All-to-all recurrent synapses (6×6)
    for src_i, src in enumerate(pop_names):
        for tgt_j, tgt in enumerate(pop_names):
            n_pre = 1
            n_post = 1
            pre_type = 'Pyramidal' if src_i < 4 else (
                'Basket' if src == 'Basket' else 'Axoaxonic'
            )
            post_type = 'Pyramidal' if tgt_j < 4 else (
                'Basket' if tgt == 'Basket' else 'Axoaxonic'
            )
            conn_key = f'{pre_type}→{post_type}'
            ts_params = _make_ts_params_dict(conn_key, n_pre, n_post)
            syn = TsodyksMarkramSynapse(
                n_pre=n_pre, n_post=n_post, dt=dt,
                params=ts_params,
                name=f'syn_rec_{src}_to_{tgt}',
            )
            graph.add_synapse(
                name=f'syn_rec_{src}_to_{tgt}',
                model=syn, src=src, tgt=tgt)

    # Input-to-population synapses
    for inp_name in input_names:
        n_pre = 2 if inp_name == 'hd' else 1
        for tgt_j, tgt in enumerate(pop_names):
            n_post = 1
            post_type = 'Pyramidal' if tgt_j < 4 else (
                'Basket' if tgt == 'Basket' else 'Axoaxonic'
            )
            conn_key = f'Input→{post_type}'
            ts_params = _make_ts_params_dict(conn_key, n_pre, n_post)
            syn = TsodyksMarkramSynapse(
                n_pre=n_pre, n_post=n_post, dt=dt,
                params=ts_params,
                name=f'syn_inp_{inp_name}_to_{tgt}',
            )
            graph.add_synapse(
                name=f'syn_inp_{inp_name}_to_{tgt}',
                model=syn, src=inp_name, tgt=tgt)

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
        firing_rates: dict of population firing rates, each [batch, n_steps, 1]
        targets: target rates [batch, n_steps, 4] for [N, S, E, W]

    Returns:
        total_loss: scalar
    """
    border_names = ['Border_N', 'Border_S', 'Border_E', 'Border_W']

    # Gather predicted border rates → [batch, n_steps, 4]
    preds_list = []
    for name in border_names:
        fr = firing_rates.get(name)
        if fr is None:
            fr = tf.zeros_like(targets[..., 0:1])
        else:
            fr = tf.squeeze(fr, axis=-1)
        preds_list.append(fr)
    preds = tf.stack(preds_list, axis=-1)  # [batch, n_steps, 4]

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
        self.traj_gen = TrajectoryGenerator(seed=config.RANDOM_SEED)

        self._use_hdf5 = os.path.exists(config.TRAJECTORY_HDF5)

        self.loss_history = []
        self.mse_history = []
        self.fr_history = []
        self.sp_history = []

        os.makedirs(config.RESULTS_DIR, exist_ok=True)

    def _get_batch(self, duration: float) -> dict:
        """Get a training batch from HDF5 or generate on the fly."""
        from utils.dataset import prepare_batch
        if self._use_hdf5:
            return prepare_batch(gen=None, duration=duration, batch_size=config.BATCH_SIZE)
        else:
            return prepare_batch(self.traj_gen, duration, config.BATCH_SIZE)

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

    def train(self, n_trials: int = None, trial_duration: float = None):
        """Run full training loop."""
        n_trials = n_trials or config.N_TRIALS
        trial_duration = trial_duration or config.TRIAL_DURATION

        for trial in range(n_trials):
            batch = self._get_batch(trial_duration)
            metrics = self.train_step(
                batch['t_seq'], batch['extra_seq'], batch['targets']
            )

            self.loss_history.append(float(metrics['loss']))
            self.mse_history.append(float(metrics['mse']))
            self.fr_history.append(float(metrics['l_fr']))
            self.sp_history.append(float(metrics['l_sp']))

            if (trial + 1) % config.PRINT_EVERY_N_TRIALS == 0:
                print(f"Trial {trial+1:4d}/{n_trials} | "
                      f"loss={metrics['loss']:.4f} | "
                      f"mse={metrics['mse']:.4f} | "
                      f"fr={metrics['l_fr']:.4f} | "
                      f"sp={metrics['l_sp']:.4f}")

            if (trial + 1) % config.SAVE_EVERY_N_TRIALS == 0:
                self.save_checkpoint(trial + 1)

        print("Training complete.")
        self.save_results()
        return self.loss_history

    def simulate(self, t_seq: np.ndarray, extra_seq: np.ndarray) -> dict:
        """Run forward simulation without training."""
        t_tensor = tf.constant(t_seq, dtype=tf.float32)
        extra_tensor = tf.constant(extra_seq, dtype=tf.float32)
        output = self.net_rnn(t_tensor, extra_inputs_seq=extra_tensor, training=False)
        result = {}
        for name in self.graph.population_names:
            if name in output.firing_rates:
                result[name] = output.firing_rates[name].numpy()
        return result

    def save_checkpoint(self, trial: int):
        """Save intermediate results."""
        path = os.path.join(config.RESULTS_DIR, f'checkpoint_trial_{trial}.npz')
        np.savez(path,
                 loss_history=self.loss_history,
                 mse_history=self.mse_history,
                 fr_history=self.fr_history,
                 sp_history=self.sp_history)

    def save_results(self):
        """Save final results."""
        path = os.path.join(config.RESULTS_DIR, 'results.npz')
        np.savez(path,
                 loss_history=self.loss_history,
                 mse_history=self.mse_history,
                 fr_history=self.fr_history,
                 sp_history=self.sp_history)

        trainable_params = self.net_rnn.trainable_variables
        param_dict = {}
        for v in trainable_params:
            param_dict[v.name] = v.numpy()
        param_path = os.path.join(config.RESULTS_DIR, 'trained_params.npz')
        np.savez(param_path, **param_dict)

        print(f"Results saved to {config.RESULTS_DIR}/")


def main():
    use_hdf5 = os.path.exists(config.TRAJECTORY_HDF5)
    print(f"Building border cell network ({config.GRAD_METHOD} mode)...")
    if use_hdf5:
        print(f"  Using trajectory from {config.TRAJECTORY_HDF5}")
    runner = SimulationRunner()
    print(f"Network built: {len(runner.graph.population_names)} populations, "
          f"{len(runner.graph.synapse_names)} synapses")
    print(f"Trainable parameters: "
          f"{np.sum([np.prod(v.shape) for v in runner.net_rnn.trainable_variables])}")
    print(f"Training for {config.N_TRIALS} trials × {config.TRIAL_DURATION}s...")
    runner.train()

    print("Running final simulation...")
    batch = runner._get_batch(config.TRIAL_DURATION * 2)
    rates = runner.simulate(batch['t_seq'], batch['extra_seq'])
    np.savez(os.path.join(config.RESULTS_DIR, 'final_simulation.npz'),
             **rates, t_seq=batch['t_seq'], traj_arr=batch['traj'],
             targets=batch['targets'])
    print("Done.")


if __name__ == '__main__':
    main()
