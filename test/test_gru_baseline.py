"""Sanity check: train a GRU network on the dataset.

If a standard GRU can learn the mapping inputs→targets,
the dataset is valid and the problem is in the biophysical model.

Usage:
    python test_gru_baseline.py [--dataset data/dataset.h5] [--epochs 50]
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import GRU, Dense
class NaNStopping(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        loss = logs.get('loss')
        if loss is None or not np.isfinite(loss):
            self.model.stop_training = True

import config
from utils.dataset import load_dataset_hdf5


def load_all_batches(dataset_path):
    ds = load_dataset_hdf5(dataset_path)
    n_batches = ds['n_batches']
    X_list, Y_list = [], []
    for i in range(n_batches):
        b = ds['get_batch'](i)
        X_list.append(b['inputs'])
        Y_list.append(b['targets'])
    ds['file'].close()
    X = np.concat(X_list).astype(np.float32)
    Y = np.concat(Y_list).astype(np.float32)
    return X, Y


def normalize(X, Y):
    """Per-channel min-max normalization to [0, 1]."""
    x_min = X.min(axis=(0, 1), keepdims=True)
    x_max = X.max(axis=(0, 1), keepdims=True)
    X_norm = (X - x_min) / (x_max - x_min + 1e-8)

    y_min = Y.min(axis=(0, 1), keepdims=True)
    y_max = Y.max(axis=(0, 1), keepdims=True)
    Y_norm = (Y - y_min) / (y_max - y_min + 1e-8)

    return X_norm, Y_norm, x_min, x_max, y_min, y_max


def build_gru(n_inputs, n_outputs, batch_size=1):
    inputs = Input(shape=(None, n_inputs), batch_size=batch_size)
    x = GRU(16, return_sequences=True, name='gru_1')(inputs)
    x = GRU(16, return_sequences=True, name='gru_2')(x)
    out = Dense(n_outputs, activation='sigmoid')(x)
    model = Model(inputs, out)
    model.compile(optimizer='adam', loss='mse')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=500)
    args = parser.parse_args()

    ds_path = args.dataset or os.path.join(
        os.path.dirname(config.TRAJECTORY_HDF5), 'dataset.h5')

    print(f"Loading dataset from {ds_path}...")
    X, Y = load_all_batches(ds_path)
    print(f"  X: {X.shape}, Y: {Y.shape}")

    X_norm, Y_norm, x_min, x_max, y_min, y_max = normalize(X, Y)
    print(f"  X range: [{X.min():.2f}, {X.max():.2f}] → [{X_norm.min():.2f}, {X_norm.max():.2f}]")
    print(f"  Y range: [{Y.min():.2f}, {Y.max():.2f}] → [{Y_norm.min():.2f}, {Y_norm.max():.2f}]")

    print("Building GRU model...")
    model = build_gru(config.N_INPUTS, n_outputs=4, batch_size=X.shape[0])
    model.summary()

    print(f"\nTraining {args.epochs} epochs...")
    t0 = time.time()
    history = model.fit(
        X_norm, Y_norm,
        epochs=args.epochs,
        batch_size=X.shape[0],
        verbose=2,
        callbacks=[NaNStopping()],
    )
    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s ({dt/args.epochs:.1f}s/epoch)")

    model.save('gru_baseline.keras')

    final_loss = history.history['loss'][-1]
    best_loss = min(history.history['loss'])
    print(f"Final loss: {final_loss:.6f}, best: {best_loss:.6f}")

    # Compute denormalized MSE for comparison with biophysical model
    Y_pred_norm = model.predict(X_norm, batch_size=X.shape[0], verbose=0)
    Y_pred = Y_pred_norm * (y_max - y_min + 1e-8) + y_min
    mse_denorm = np.mean((Y_pred - Y) ** 2)
    print(f"Denormalized MSE: {mse_denorm:.6f}")

    # Per-wall accuracy
    wall_names = ['N', 'S', 'E', 'W']
    for j in range(4):
        mse_j = np.mean((Y_pred[:, :, j] - Y[:, :, j]) ** 2)
        corr_j = np.corrcoef(Y[:, :, j].ravel(), Y_pred[:, :, j].ravel())[0, 1]
        print(f"  Wall {wall_names[j]}: MSE={mse_j:.6f}, corr={corr_j:.4f}")


if __name__ == '__main__':
    main()
