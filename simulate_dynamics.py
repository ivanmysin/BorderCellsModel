"""Run trained model on the dataset and save firing-rate dynamics.

Uses train_simple.build_model() and train_simple.load_pretrained() to
guarantee the architecture and weight loading match the trained model.
State propagates across batches (no reset), as in the training run.

Usage:
    python simulate_dynamics.py [--weights results/training.h5]
                                [--output results/dynamics.h5]
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
import train_simple


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=str, default='results/training.h5',
                        help='HDF5 with saved trainable weights (from train_simple).')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset.h5; default = alongside trajectory.h5')
    parser.add_argument('--output', type=str, default='results/dynamics.h5',
                        help='Output HDF5 path for the dynamics.')
    parser.add_argument('--start-batch', type=int, default=0,
                        help='First batch index (inclusive).')
    parser.add_argument('--end-batch', type=int, default=None,
                        help='Last batch index (exclusive). Default = all batches.')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    ds_path = args.dataset or os.path.join(
        os.path.dirname(config.TRAJECTORY_HDF5), 'dataset.h5')

    if args.seed is not None:
        config.RANDOM_SEED = args.seed
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    print("Configuring devices...")
    train_simple.setup_gpu()

    print(f"Loading dataset from {ds_path}...")
    X, Y = train_simple.load_all_batches(ds_path)
    n_batches = X.shape[0]
    T = X.shape[2]
    print(f"  n_batches={n_batches}, T={T}, F_in={X.shape[3]}")

    start = max(0, args.start_batch)
    end = args.end_batch if args.end_batch is not None else n_batches
    end = min(end, n_batches)
    n_run = end - start
    if n_run <= 0:
        raise ValueError(f"Empty range: start={start}, end={end}")
    print(f"Running on batches [{start}, {end}) = {n_run} batches")

    print("Building model via train_simple.build_model (identical to trained)...")
    model = train_simple.build_model()
    n_vars = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"  Trainable parameters: {n_vars}")

    if args.weights and os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}...")
        n_loaded = train_simple.load_pretrained(model, args.weights)
        print(f"  Loaded {n_loaded} trainable variable(s)")
    else:
        print(f"WARNING: weights '{args.weights}' not found, using initial values")

    print("Running model (state propagates across batches)...")
    rates_all = np.zeros((n_run, 1, T, config.N_POP_UNITS), dtype=np.float32)
    targets_all = np.zeros((n_run, 1, T, 4), dtype=np.float32)
    t_start = time.time()
    for i, b in enumerate(range(start, end)):
        t0 = time.time()
        pred = model.predict(X[b], verbose=0)
        rates_all[i] = pred
        targets_all[i] = Y[b]
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            print(f"  batch {b} ({i+1}/{n_run}): "
                  f"dt={time.time()-t0:.1f}s, total={elapsed:.0f}s")

    print(f"Done in {(time.time()-t_start)/60:.1f} min")
    print(f"rates shape: {rates_all.shape}, "
          f"range=[{rates_all.min():.3f}, {rates_all.max():.3f}], "
          f"mean border={rates_all[..., :4].mean():.3f}, "
          f"mean basket/axo={rates_all[..., 4:].mean():.3f}")
    print(f"targets shape: {targets_all.shape}, "
          f"range=[{targets_all.min():.3f}, {targets_all.max():.3f}]")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('rates', data=rates_all, compression='gzip')
        f.create_dataset('targets', data=targets_all, compression='gzip')
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
