"""Network-level tests: build, simulate, and plot activity.

Each test runs in a separate subprocess to isolate TF graphs.
Three scenarios:
  1. Zero synaptic conductances — isolated neuron dynamics
  2. Zero I_ext + zero synapses — generator influence (no intrinsic drive)
  3. Default settings — full network (pre-existing NaN issue)
"""

import os
import sys
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config

RESULTS_DIR = os.path.join(config.RESULTS_DIR, "tests")
os.makedirs(RESULTS_DIR, exist_ok=True)

UNIT_LABELS = config.UNIT_NAMES
COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

N_STEPS = 10000
DT_MS = config.DT


def _worker(scenario: str):
    """Run a single scenario, save plot, print result. Called in subprocess."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    from simulate import build_network, build_integrator
    from neuraltide.core.network import NetworkRNN
    from utils.trajectory import TrajectoryGenerator
    from utils.dataset import prepare_batch

    g = build_network()
    pop = g._populations[config.POPULATION_NAME]
    pop.dt = DT_MS

    if scenario == 'zero_syn':
        for syn_name in g.synapse_names:
            g._synapses[syn_name].model.gsyn_max.assign(
                np.zeros_like(g._synapses[syn_name].model.gsyn_max.numpy()))
    elif scenario == 'zero_iext':
        pop.I_ext.assign(np.zeros([config.N_UNITS]))

        # for syn_name in g.synapse_names:
        #     g._synapses[syn_name].model.gsyn_max.assign(
        #         np.zeros_like(g._synapses[syn_name].model.gsyn_max.numpy()))

    integrator = build_integrator()
    net = NetworkRNN(g, integrator, return_hidden_states=False)

    gen = TrajectoryGenerator(seed=42)
    duration_ratinabox = N_STEPS * DT_MS / 1000.0 + config.TRAJECTORY_DT
    batch = prepare_batch(gen, duration=duration_ratinabox, batch_size=1)

    t_seq = (np.arange(N_STEPS, dtype=np.float32) * DT_MS).reshape(1, -1, 1)
    extra_raw = batch['extra_seq']
    t_indices = np.linspace(0, extra_raw.shape[1] - 1, N_STEPS).astype(int)
    t_indices = np.clip(t_indices, 0, extra_raw.shape[1] - 1)
    extra_seq = extra_raw[:, t_indices, :]

    output = net(tf.constant(t_seq), tf.constant(extra_seq), training=False)
    rates = output.firing_rates[config.POPULATION_NAME].numpy()

    times = np.arange(N_STEPS) * DT_MS / 1000.0
    has_nan = np.any(np.isnan(rates))
    nan_step = -1
    if has_nan:
        nan_positions = np.where(np.any(np.isnan(rates[0]), axis=1))[0]
        nan_step = nan_positions[0] if len(nan_positions) > 0 else -1

    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    for i, (label, color) in enumerate(zip(UNIT_LABELS, COLORS)):
        ax = axes[i]
        ax.plot(times[:nan_step if has_nan else len(times)],
                rates[0, :nan_step if has_nan else len(times), i],
                color=color, linewidth=0.8)
        ax.set_ylabel(f'{label}\n(Hz)', fontsize=8)
        ax.tick_params(labelsize=7)
        if has_nan:
            ax.axvline(x=times[nan_step], color='red', linestyle='--',
                       linewidth=0.5, alpha=0.5)
    axes[0].set_title({
        'zero_syn': 'Zero Synaptic Conductances — Isolated Neuron Dynamics',
        'zero_iext': 'Zero I_ext + Zero Synapses — No Drive (all flat)',
        'default': 'Default Settings — Full Network (pre-existing NaN)',
    }[scenario], fontsize=11)
    axes[-1].set_xlabel('Time (s)', fontsize=9)
    if has_nan:
        fig.text(0.5, 0.01, f'NaN at t={times[nan_step]:.4f}s (red)',
                 ha='center', fontsize=9, color='red')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(RESULTS_DIR, f'test_network_{scenario}.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)

    print(f'rates [{np.nanmin(rates):.3f}, {np.nanmax(rates):.3f}] '
          f'NaN={has_nan}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('zero_syn', 'zero_iext', 'default'):
        _worker(sys.argv[1])
    else:
        print('Running network tests (3 subprocesses)...')
        for scenario in ['zero_syn', 'zero_iext', 'default']:
            label = {'zero_syn': 'Zero Synapses',
                     'zero_iext': 'Zero I_ext',
                     'default': 'Default'}[scenario]
            print(f'  {label}...', end=' ', flush=True)
            result = subprocess.run(
                [sys.executable, __file__, scenario],
                capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print(f'FAIL ({result.returncode})')
                print(result.stderr)
        print('All network tests completed.')
