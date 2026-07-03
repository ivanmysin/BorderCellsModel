"""Train border cell network with adjoint state method.

Usage:
    python train.py [--dataset data/dataset.h5] [--epochs 100] [--lr 1e-3]
"""

import os
import argparse
import json

import numpy as np
import tensorflow as tf

import config
import neuraltide as nt
from neuraltide.core.network import NetworkGraph
from neuraltide.model import BrainModelKeras
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.integrators import RK4Integrator
from neuraltide.training import DivergenceDetector
from neuraltide.training.adjoint import AdjointSolver
from neuraltide.training.losses import MSELoss, CompositeLoss, StabilityPenalty

from utils.dataset import load_dataset_hdf5
from utils.params import (
    build_pop_params,
    build_inp_gsyn_matrix, build_inp_tau_f_matrix, build_inp_tau_d_matrix,
    build_inp_tau_r_matrix, build_inp_Uinc_matrix, build_inp_pconn_matrix,
    build_inp_e_r_matrix,
    build_rec_gsyn_matrix, build_rec_tau_f_matrix, build_rec_tau_d_matrix,
    build_rec_tau_r_matrix, build_rec_Uinc_matrix, build_rec_pconn_matrix,
    build_rec_e_r_matrix,
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

_MSE_WEIGHT = 1.0
_STAB_WEIGHT = 1e-3


@tf.function
def _adjoint_step(solver, optimizer, t_seq, inputs, target_padded, stab_weight):
    """One adjoint training step, fully compiled with @tf.function.

    Target is passed as a tensor so the trace is shared across batches.
    """
    loss_fn = CompositeLoss([
        (_MSE_WEIGHT, MSELoss(target={config.POPULATION_NAME: target_padded})),
        (stab_weight, StabilityPenalty()),
    ])
    grads, variables, output = solver.compute_gradients(
        t_seq, inputs, {config.POPULATION_NAME: target_padded}, loss_fn,
    )
    loss = loss_fn(output, solver._network)

    grads_and_vars = [
        (g, v) for g, v in zip(grads, variables) if g is not None
    ]
    if grads_and_vars:
        grads_only = [g for g, _ in grads_and_vars]
        clipped, _ = tf.clip_by_global_norm(grads_only, 1.0)
        clipped_gv = [(c, v) for (_, v), c in zip(grads_and_vars, clipped)]
        optimizer.apply_gradients(clipped_gv)
    return loss


def build_network() -> BrainModelKeras:
    """Build the vectorized border cell network as BrainModelKeras."""
    dt = config.DT
    graph = NetworkGraph(dt=dt)

    graph.declare_input('inputs', n_units=config.N_INPUTS)

    pop_params = build_pop_params()
    pop = IzhikevichMeanField(dt=dt, params={
        'tau_pop':  pop_params['tau_pop'],
        'alpha':    pop_params['alpha'],
        'a':        pop_params['a'],
        'b':        pop_params['b'],
        'w_jump':   pop_params['w_jump'],
        'Delta_I':  {
            'value': pop_params['Delta_I'],
            'trainable': config.TRAIN_POP_DELTA_I,
            'min': 0.00001, 'max': 0.5,
        },
        'I_ext':    {
            'value': pop_params['I_ext'],
            'trainable': config.TRAIN_POP_IEXT,
        },
    }, name=config.POPULATION_NAME)
    graph.add_population(config.POPULATION_NAME, pop)

    syn_in = TsodyksMarkramSynapse(
        n_pre=config.N_INPUTS, n_post=config.N_POP_UNITS, dt=dt, params={
            'gsyn_max': {
                'value': build_inp_gsyn_matrix(),
                'trainable': config.TRAIN_SYNAPSE_GMAX,
                'min': 0.0,
            },
            'tau_f': {
                'value': build_inp_tau_f_matrix(),
                'trainable': config.TRAIN_SYNAPSE_TAU_f,
                'min': 6.0, 'max': 240.0,
            },
            'tau_d': {
                'value': build_inp_tau_d_matrix(),
                'trainable': config.TRAIN_SYNAPSE_TAU_d,
                'min': 2.0, 'max': 15.0,
            },
            'tau_r': {
                'value': build_inp_tau_r_matrix(),
                'trainable': config.TRAIN_SYNAPSE_TAU_r,
                'min': 91.0, 'max': 1300.0,
            },
            'Uinc': {
                'value': build_inp_Uinc_matrix(),
                'trainable': config.TRAIN_SYNAPSE_U,
                'min': 0.1, 'max': 0.7,
            },
            'pconn': {
                'value': build_inp_pconn_matrix(),
                'trainable': False,
            },
            'e_r': {
                'value': build_inp_e_r_matrix(),
                'trainable': False,
            },
        }, name='syn_inp')
    graph.add_synapse('inp->cells', syn_in, src='inputs', tgt=config.POPULATION_NAME)

    syn_rec = TsodyksMarkramSynapse(
        n_pre=config.N_POP_UNITS, n_post=config.N_POP_UNITS, dt=dt, params={
            'gsyn_max': {
                'value': build_rec_gsyn_matrix(),
                'trainable': config.TRAIN_SYNAPSE_GMAX,
                'min': 0.0,
            },
            'tau_f': {
                'value': build_rec_tau_f_matrix(),
                'trainable': config.TRAIN_SYNAPSE_TAU_f,
                'min': 6.0, 'max': 240.0,
            },
            'tau_d': {
                'value': build_rec_tau_d_matrix(),
                'trainable': config.TRAIN_SYNAPSE_TAU_d,
                'min': 2.0, 'max': 15.0,
            },
            'tau_r': {
                'value': build_rec_tau_r_matrix(),
                'trainable': config.TRAIN_SYNAPSE_TAU_r,
                'min': 91.0, 'max': 1300.0,
            },
            'Uinc': {
                'value': build_rec_Uinc_matrix(),
                'trainable': config.TRAIN_SYNAPSE_U,
                'min': 0.1, 'max': 0.7,
            },
            'pconn': {
                'value': build_rec_pconn_matrix(),
                'trainable': False,
            },
            'e_r': {
                'value': build_rec_e_r_matrix(),
                'trainable': False,
            },
        }, name='syn_rec')
    graph.add_synapse('cells->cells', syn_rec,
                      src=config.POPULATION_NAME, tgt=config.POPULATION_NAME)

    graph.validate()
    return BrainModelKeras(
        graph,
        integrator=RK4Integrator(),
        dt=dt,
        stability_penalty_weight=1e-3,
    )


def train(dataset_path: str = None, n_epochs: int = None,
          learning_rate: float = None, batches_per_epoch: int = None):
    """Run training loop over dataset batches using adjoint state method.

    Each epoch samples `batches_per_epoch` (default 50) random batches
    without replacement from the dataset.
    """

    ds_path = dataset_path or os.path.join(
        os.path.dirname(config.TRAJECTORY_HDF5), 'dataset.h5')
    n_epochs = n_epochs or config.N_EPOCHS
    lr = learning_rate or config.LEARNING_RATE


    print(f"Loading dataset from {ds_path}...")
    ds = load_dataset_hdf5(ds_path)
    n_batches = ds['n_batches']
    print(f"  {n_batches} batches available, batch_steps={ds['metadata']['batch_steps']}")

    tf.random.set_seed(config.RANDOM_SEED)
    nt.seed_everything(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("Building network...")
    model = build_network()
    network = model.network
    n_vars = sum(np.prod(v.shape) for v in network.trainable_variables)
    print(f"  Trainable parameters: {int(n_vars)}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    solver = AdjointSolver(network, network._integrator)

    divergence_detector = DivergenceDetector()

    loss_history = []
    best_loss = float('inf')

    print(f"Training (adjoint): {n_epochs} epochs x {batches_per_epoch} "
          f"random batches/epoch (out of {n_batches})...")
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_finite = 0
        batch_indices = np.random.choice(
            n_batches, size=batches_per_epoch, replace=False)

        for batch_idx in batch_indices:
            batch = ds['get_batch'](int(batch_idx))
            t_seq = tf.constant(batch['t_seq'], dtype=tf.float32)
            inputs = tf.constant(batch['inputs'], dtype=tf.float32)
            target_4 = tf.constant(batch['targets'], dtype=tf.float32)
            target_padded = tf.pad(target_4, [[0, 0], [0, 0], [0, 2]])

            loss = _adjoint_step(
                solver, optimizer, t_seq, inputs, target_padded, _STAB_WEIGHT,
            )
            loss_val = float(loss)
            if np.isfinite(loss_val):
                epoch_loss += loss_val
                n_finite += 1

        avg_loss = epoch_loss / max(n_finite, 1)
        loss_history.append(float(avg_loss))

        if avg_loss < best_loss:
            best_loss = avg_loss

        if hasattr(divergence_detector, 'on_epoch_end'):
            divergence_detector.on_epoch_end(epoch, {'loss': avg_loss})

        if (epoch + 1) % max(1, n_epochs // 20) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs} | loss={avg_loss:.6f}"
                  f" | best={best_loss:.6f} | finite_batches={n_finite}/{batches_per_epoch}")

    ds['file'].close()

    print("Saving results...")
    save_training_results(loss_history, network)

    return loss_history


def save_training_results(loss_history, network):
    """Save training results to HDF5 and JSON."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    h5_path = os.path.join(config.RESULTS_DIR, 'training.h5')
    import h5py
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('loss_history', data=np.array(loss_history))

        grp = f.create_group('parameters')
        used_names = set()
        for idx, v in enumerate(network.trainable_variables):
            base = v.name.replace(':', '_').replace('/', '_') or f'param_{idx}'
            name = base
            suffix = 0
            while name in used_names:
                suffix += 1
                name = f'{base}_{suffix}'
            used_names.add(name)
            grp.create_dataset(name, data=v.numpy())

        cfg = f.create_group('config')
        for attr in dir(config):
            if attr.isupper() and not attr.startswith('_'):
                val = getattr(config, attr)
                if isinstance(val, (int, float, str, bool)):
                    cfg.attrs[attr] = val

    print(f"  HDF5 results saved to {h5_path}")

    json_path = os.path.join(config.RESULTS_DIR, 'training.json')
    payload = {
        'loss_history': [float(x) for x in loss_history],
        'epochs': len(loss_history),
        'grad_method': config.GRAD_METHOD,
        'n_batches_per_epoch': config.N_BATCHES_PER_EPOCH,
        'trainable_variables': [
            {'name': v.name, 'value': v.numpy().tolist()}
            for v in network.trainable_variables
        ],
    }
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON results saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Train border cell network")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to dataset HDF5")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--batches-per-epoch", type=int, default=None,
                        help="Number of random batches per epoch (default 50)")
    args = parser.parse_args()

    if args.seed is not None:
        config.RANDOM_SEED = args.seed

    train(args.dataset, args.epochs, args.lr, args.batches_per_epoch)


if __name__ == '__main__':
    main()
