"""Train border cell network using neuraltide tools (adjoint state method).

Usage:
    python train.py [--dataset data/dataset.h5] [--epochs 100] [--lr 1e-3]
"""

import os
import argparse

import numpy as np
import tensorflow as tf

import config
import neuraltide as nt
from neuraltide.core.network import NetworkGraph, NetworkRNN
from neuraltide.populations import IzhikevichMeanField
from neuraltide.synapses import TsodyksMarkramSynapse
from neuraltide.integrators import RK4Integrator
from neuraltide.training import (
    Trainer,
    CompositeLoss,
    MSELoss,
    StabilityPenalty,
    DivergenceDetector,
)

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


def build_network() -> NetworkRNN:
    """Build the vectorized border cell network."""
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
    return NetworkRNN(
        graph,
        integrator=RK4Integrator(),
        stability_penalty_weight=1e-3,
    )


def _batch_target_dict(targets_4d: np.ndarray) -> dict:
    """Pad target rates [B,T,4] → [B,T,6] (Basket / Axo targeted to 0 Hz)."""
    targets_4 = tf.constant(targets_4d, dtype=tf.float32)
    targets_6 = tf.pad(targets_4, [[0, 0], [0, 0], [0, 2]])
    return {config.POPULATION_NAME: targets_6}


def _batch_loss(target_dict: dict) -> CompositeLoss:
    """Build CompositeLoss for the current batch's targets."""
    return CompositeLoss([
        (1.0, MSELoss(target_dict)),
        (1e-3, StabilityPenalty()),
    ])


def train(dataset_path: str = None, n_epochs: int = None,
          learning_rate: float = None):
    """Run training loop over dataset batches using adjoint state method."""

    ds_path = dataset_path or os.path.join(
        os.path.dirname(config.TRAJECTORY_HDF5), 'dataset.h5')
    n_epochs = n_epochs or config.N_EPOCHS
    lr = learning_rate or config.LEARNING_RATE

    print(f"Loading dataset from {ds_path}...")
    ds = load_dataset_hdf5(ds_path)
    n_batches = ds['n_batches']
    print(f"  {n_batches} batches, batch_steps={ds['metadata']['batch_steps']}")

    tf.random.set_seed(config.RANDOM_SEED)
    nt.seed_everything(config.RANDOM_SEED)

    print("Building network...")
    network = build_network()
    n_vars = sum(np.prod(v.shape) for v in network.trainable_variables)
    print(f"  Trainable parameters: {int(n_vars)}")

    placeholder_target = {
        config.POPULATION_NAME: tf.zeros([1, 1, config.N_POP_UNITS], dtype=tf.float32)
    }
    trainer = Trainer(
        network=network,
        loss_fn=_batch_loss(placeholder_target),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        grad_method='adjoint',
        grad_clip_norm=1.0,
    )

    divergence_detector = DivergenceDetector()

    loss_history = []
    best_loss = float('inf')

    print(f"Training (adjoint state method): "
          f"{n_epochs} epochs x {n_batches} batches...")
    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for batch_idx in range(n_batches):
            batch = ds['get_batch'](batch_idx)
            t_seq = tf.constant(batch['t_seq'])
            inputs = network._graph.pack_inputs({
                'inputs': tf.constant(batch['inputs'], dtype=tf.float32)
            })

            target_dict = _batch_target_dict(batch['targets'])
            trainer.loss_fn = _batch_loss(target_dict)

            step = trainer.train_step(t_seq, inputs)
            epoch_loss += float(step['loss'])

        avg_loss = epoch_loss / n_batches
        loss_history.append(float(avg_loss))

        if avg_loss < best_loss:
            best_loss = avg_loss

        if hasattr(divergence_detector, 'on_epoch_end'):
            divergence_detector.on_epoch_end(epoch, {'loss': avg_loss})

        if (epoch + 1) % max(1, n_epochs // 20) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{n_epochs} | loss={avg_loss:.6f}"
                  f" | best={best_loss:.6f}")

    trainer._last_history = type(
        'H', (), {'loss_history': loss_history, 'epochs': n_epochs}
    )()

    ds['file'].close()

    print("Saving results...")
    save_training_results(loss_history, network, trainer)

    return loss_history


def save_training_results(loss_history, network, trainer):
    """Save training results to HDF5 and JSON."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    json_path = os.path.join(config.RESULTS_DIR, 'training.json')
    trainer.export_results(json_path, format='json')
    print(f"  Trainer results saved to {json_path}")

    h5_path = os.path.join(config.RESULTS_DIR, 'training.h5')
    import h5py
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('loss_history', data=np.array(loss_history))

        grp = f.create_group('parameters')
        for v in network.trainable_variables:
            name = v.name.replace(':', '_').replace('/', '_')
            grp.create_dataset(name, data=v.numpy())

        cfg = f.create_group('config')
        for attr in dir(config):
            if attr.isupper() and not attr.startswith('_'):
                val = getattr(config, attr)
                if isinstance(val, (int, float, str, bool)):
                    cfg.attrs[attr] = val

    print(f"  HDF5 results saved to {h5_path}")


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
    args = parser.parse_args()

    if args.seed is not None:
        config.RANDOM_SEED = args.seed

    train(args.dataset, args.epochs, args.lr)


if __name__ == '__main__':
    main()
