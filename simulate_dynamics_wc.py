"""Run trained Wilson-Cowan model on the dataset and save dynamics.

Usage:
    python simulate_dynamics_wc.py [--weights results/checkpoints_wc/latest.weights.h5]
                                   [--output results/dynamics_wc.h5]
                                   [--start-batch 0] [--end-batch N]
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import time

import numpy as np
import h5py
import tensorflow as tf

import config
import train_wc
from utils.dataset import load_dataset_hdf5


def load_weights(model, path):
    if not os.path.exists(path):
        print(f"  WARNING: '{path}' not found, using initial values")
        return
    print(f"Loading weights from {path}...")
    model.load_weights(path)
    print(f"  Loaded (checkpoint format)")


def load_all_batches(dataset_path):
    ds = load_dataset_hdf5(dataset_path)
    n_batches = ds['n_batches']
    print(f"  Loading {n_batches} batches into RAM...")
    X_list, Y_list = [], []
    for i in range(n_batches):
        b = ds['get_batch'](i)
        X_list.append(b['inputs'])
        Y_list.append(b['targets'])
    ds['file'].close()
    X = np.concat(X_list).astype(np.float32)
    Y = np.concat(Y_list).astype(np.float32)
    print(f"  X: {X.shape}, Y: {Y.shape}, {X.nbytes / 1e6:.1f} MB")
    return X, Y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=str,
                        default='results/checkpoints_wc/latest.weights.h5')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--output', type=str, default='results/dynamics_wc.h5')
    parser.add_argument('--start-batch', type=int, default=0)
    parser.add_argument('--end-batch', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--layers', type=int, default=1)
    args = parser.parse_args()

    ds_path = args.dataset or os.path.join(
        os.path.dirname(config.TRAJECTORY_HDF5), 'dataset.h5')

    if args.seed is not None:
        config.RANDOM_SEED = args.seed
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    gpus = tf.config.list_physical_devices('GPU')
    print(f"  Devices: {len(gpus)} GPU(s)")

    print(f"Loading dataset from {ds_path}...")
    X, Y = load_all_batches(ds_path)
    n_batches = X.shape[0]
    T = X.shape[1]

    start = max(0, args.start_batch)
    end = args.end_batch if args.end_batch is not None else n_batches
    end = min(end, n_batches)
    n_run = end - start
    if n_run <= 0:
        raise ValueError(f"Empty range: start={start}, end={end}")
    print(f"Running on batches [{start}, {end}) = {n_run} batches")

    print(f"Building Wilson-Cowan model ({args.layers} layers)...")
    model = train_wc.build_model(n_layers=args.layers)
    n_vars = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"  Trainable parameters: {n_vars}")

    load_weights(model, args.weights)

    print("Running model (state propagates across batches)...")
    rates_all = np.zeros((n_run, T, config.N_POP_UNITS), dtype=np.float32)
    targets_all = np.zeros((n_run, T, 4), dtype=np.float32)
    t_start = time.time()
    for i, b in enumerate(range(start, end)):
        t0 = time.time()
        pred = model.predict(X[b:b+1], verbose=0)
        rates_all[i] = np.asarray(pred)
        targets_all[i] = Y[b]
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            print(f"  batch {b} ({i+1}/{n_run}): "
                  f"dt={time.time()-t0:.1f}s, total={elapsed:.0f}s")

    print(f"Done in {(time.time()-t_start)/60:.1f} min")
    print(f"rates: {rates_all.shape}, "
          f"[{rates_all.min():.3f}, {rates_all.max():.3f}], "
          f"mean border={rates_all[..., :4].mean():.3f}")
    print(f"targets: {targets_all.shape}, "
          f"[{targets_all.min():.3f}, {targets_all.max():.3f}]")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('rates', data=rates_all)
        f.create_dataset('targets', data=targets_all)
        f.attrs['start_batch'] = start
        f.attrs['end_batch'] = end
        f.attrs['n_batches'] = n_run
        f.attrs['T'] = T
        f.attrs['dt'] = config.DT
        f.attrs['arena_cm'] = config.ARENA_CM
        f.attrs['unit_names'] = np.array(config.UNIT_NAMES, dtype='S')
        f.attrs['weights_source'] = args.weights or 'init'
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
