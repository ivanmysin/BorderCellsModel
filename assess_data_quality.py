#!/usr/bin/env python3
"""Оценка качества сгенерированных данных: проверка наличия X -> Y связи.

X = firing rates 21 входных нейронов, Y = firing rates 4 целевых нейронов.
Метод: HistGradientBoostingRegressor per (target, lag). R^2 heatmap,
permutation test, feature importance.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
import json
import sys
import time
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error


DEFAULT_DATASET = 'data/dataset.h5'
DEFAULT_RESULTS = 'results/assessment'
DEFAULT_LAGS = [0, 5, 10, 15, 20, 25, 30, 40, 50]
TARGET_NAMES = [f'y{j}' for j in range(4)]
INPUT_NAMES = [f'x{i}' for i in range(21)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default=DEFAULT_DATASET)
    p.add_argument('--results_dir', default=DEFAULT_RESULTS)
    p.add_argument('--n_samples', type=int, default=500_000,
                   help='Число случайных точек для оценки (default 500000).')
    p.add_argument('--lags', type=int, nargs='+', default=DEFAULT_LAGS,
                   help='Сетка лагов в шагах dt.')
    p.add_argument('--max_iter', type=int, default=100)
    p.add_argument('--max_depth', type=int, default=6)
    p.add_argument('--learning_rate', type=float, default=0.05)
    p.add_argument('--n_permutations', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--dry_run', action='store_true',
                   help='Использовать 50k точек и пропустить permutation test.')
    return p.parse_args()


def load_random_chunks(path: str, n_samples: int, rng: np.random.Generator):
    """Случайная стратифицированная подвыборка: идём по батчам в случайном порядке,
    из каждого берём k строк, пока не набрали n_samples.

    Структура HDF5: dataset/batch_{i}/{t_seq, inputs, targets}, inputs = (1, T, 21).
    """
    samples_x: list[np.ndarray] = []
    samples_y: list[np.ndarray] = []
    remaining = n_samples
    with h5py.File(path, 'r') as f:
        root = f['dataset']
        batch_names = sorted(
            [k for k in root.keys() if k.startswith('batch_')],
            key=lambda s: int(s.split('_')[1]),
        )
        n_batches = len(batch_names)
        n_steps = root[batch_names[0]]['inputs'].shape[1]
        order = rng.permutation(batch_names)
        per_batch = int(np.ceil(n_samples / n_batches)) + 1
        for bn in order:
            if remaining <= 0:
                break
            grp = root[bn]
            take = min(per_batch, n_steps, remaining)
            t_idx = rng.choice(n_steps, size=take, replace=False)
            t_idx.sort()
            xi = grp['inputs'][0, t_idx, :].astype(np.float32)  # (take, 21)
            yi = grp['targets'][0, t_idx, :].astype(np.float32)  # (take, 4)
            samples_x.append(xi)
            samples_y.append(yi)
            remaining -= take
    X = np.concatenate(samples_x, axis=0)[:n_samples]
    Y = np.concatenate(samples_y, axis=0)[:n_samples]
    return X, Y


def evaluate_grid(X: np.ndarray, Y: np.ndarray, lags, args, grid_path: str):
    """Обучить GBT per (target, lag); поддержка resume через npz на диске."""
    n_targets = Y.shape[1]
    n_lags = len(lags)
    r2_grid = np.full((n_targets, n_lags), np.nan)
    mae_grid = np.full((n_targets, n_lags), np.nan)

    # resume
    if os.path.exists(grid_path):
        try:
            saved = np.load(grid_path)
            r2_grid = saved['r2']
            mae_grid = saved['mae']
            print(f'[resume] загружено {np.sum(~np.isnan(r2_grid))} готовых пар', flush=True)
        except Exception as e:
            print(f'[resume] не удалось: {e}', flush=True)

    total = n_targets * n_lags
    done = 0
    t_start = time.time()
    for j in range(n_targets):
        for li, k in enumerate(lags):
            if not np.isnan(r2_grid[j, li]):
                continue
            done += 1
            t0 = time.time()
            # Y_t = f(X_{t-k}); сдвигаем X на k назад во времени
            if k == 0:
                X_lag, y = X, Y[:, j]
            else:
                X_lag, y = X[:-k], Y[k:, j]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_lag, y, test_size=args.test_size, random_state=args.seed
            )
            model = HistGradientBoostingRegressor(
                max_iter=args.max_iter,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                random_state=args.seed,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            r2 = float(r2_score(y_te, y_pred))
            mae = float(mean_absolute_error(y_te, y_pred))
            r2_grid[j, li] = r2
            mae_grid[j, li] = mae
            dt = time.time() - t0
            elapsed = time.time() - t_start
            eta = (elapsed / done) * (total - np.sum(~np.isnan(r2_grid)))
            print(
                f'[{done:2d}/{total}] target=y{j} lag={k:2d}  R^2={r2:+.4f}  '
                f'MAE={mae:.4f}  ({dt:5.1f}s, eta={eta:5.0f}s)',
                flush=True,
            )
            np.savez(grid_path, r2=r2_grid, mae=mae_grid,
                     lags=np.array(lags), n_samples=X.shape[0])
    return r2_grid, mae_grid


def permutation_test(X: np.ndarray, y: np.ndarray, real_r2: float,
                     n_perm: int, args, rng: np.random.Generator) -> dict:
    """Перемешать Y, обучить GBT n_perm раз, собрать null R^2."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    null_r2 = np.empty(n_perm, dtype=np.float32)
    for i in range(n_perm):
        y_tr_perm = rng.permutation(y_tr)
        model = HistGradientBoostingRegressor(
            max_iter=args.max_iter,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        model.fit(X_tr, y_tr_perm)
        y_pred = model.predict(X_te)
        null_r2[i] = r2_score(y_te, y_pred)
    p_value = float((np.sum(null_r2 >= real_r2) + 1) / (n_perm + 1))
    return {
        'real_r2': float(real_r2),
        'null_mean': float(np.mean(null_r2)),
        'null_std': float(np.std(null_r2)),
        'null_r2': null_r2.tolist(),
        'p_value': p_value,
    }


def feature_importance(X: np.ndarray, y: np.ndarray, args) -> np.ndarray:
    """Permutation importance на holdout."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    model = HistGradientBoostingRegressor(
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(X_tr, y_tr)
    result = permutation_importance(
        model, X_te, y_te, n_repeats=10, random_state=args.seed, n_jobs=1
    )
    return result.importances_mean


def plot_heatmap(r2: np.ndarray, lags, path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(r2, aspect='auto', cmap='RdYlGn', vmin=-0.1, vmax=1.0)
    ax.set_yticks(range(len(TARGET_NAMES)))
    ax.set_yticklabels(TARGET_NAMES)
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels([f'{l}' for l in lags])
    ax.set_xlabel('Lag (шагов dt=0.1с)')
    ax.set_ylabel('Целевой нейрон')
    ax.set_title('R^2: предсказание Y по X с лагом')
    for j in range(r2.shape[0]):
        for li in range(r2.shape[1]):
            ax.text(li, j, f'{r2[j, li]:.2f}',
                    ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, label='R^2')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_feature_importance(imp: np.ndarray, path: str):
    order = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(imp)), imp[order][::-1])
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels([INPUT_NAMES[i] for i in order][::-1])
    ax.set_xlabel('Permutation importance')
    ax.set_title('Важность входных нейронов для лучшей модели')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        args.n_samples = 50_000
        args.n_permutations = 0
    os.makedirs(args.results_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f'== assess_data_quality ==')
    print(f'dataset      : {args.dataset}')
    print(f'results_dir  : {args.results_dir}')
    print(f'n_samples    : {args.n_samples}')
    print(f'lags         : {args.lags}')
    print(f'GBT params   : max_iter={args.max_iter} depth={args.max_depth} lr={args.learning_rate}')
    print(f'dry_run      : {args.dry_run}')

    print(f'\n[1/4] Загрузка {args.n_samples} точек из {args.dataset} ...', flush=True)
    t0 = time.time()
    X, Y = load_random_chunks(args.dataset, args.n_samples, rng)
    print(f'  X.shape={X.shape}  Y.shape={Y.shape}  ({time.time()-t0:.1f}s)', flush=True)
    print(f'  X stats: mean={X.mean():.3f}  std={X.std():.3f}  min={X.min():.3f}  max={X.max():.3f}')
    print(f'  Y stats: mean={Y.mean(axis=0)}')
    print(f'         std ={Y.std(axis=0)}')

    grid_path = os.path.join(args.results_dir, 'grid.npz')
    print(f'\n[2/4] Обучение GBT per (target, lag) -> {grid_path}', flush=True)
    r2_grid, mae_grid = evaluate_grid(X, Y, args.lags, args, grid_path)

    print(f'\n[3/4] Лучшая пара (target, lag) ...', flush=True)
    best_j, best_li = np.unravel_index(np.nanargmax(r2_grid), r2_grid.shape)
    best_lag = args.lags[best_li]
    best_r2 = float(r2_grid[best_j, best_li])
    best_mae = float(mae_grid[best_j, best_li])
    print(f'  target=y{best_j}  lag={best_lag} (={best_lag*0.1:.1f}с)  '
          f'R^2={best_r2:+.4f}  MAE={best_mae:.4f}')

    summary = {
        'n_samples': int(X.shape[0]),
        'lags': list(args.lags),
        'r2_grid': r2_grid.tolist(),
        'mae_grid': mae_grid.tolist(),
        'best': {
            'target': f'y{best_j}',
            'lag_steps': int(best_lag),
            'lag_seconds': float(best_lag * 0.1),
            'r2': best_r2,
            'mae': best_mae,
        },
        'gbt_params': {
            'max_iter': args.max_iter,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
        },
    }

    if args.n_permutations > 0:
        print(f'\n[3a/4] Permutation test ({args.n_permutations} перестановок) ...', flush=True)
        if best_lag == 0:
            X_lag, y_best = X, Y[:, best_j]
        else:
            X_lag, y_best = X[:-best_lag], Y[best_lag:, best_j]
        perm = permutation_test(X_lag, y_best, best_r2,
                                args.n_permutations, args, rng)
        summary['permutation_test'] = {
            k: v for k, v in perm.items() if k != 'null_r2'
        }
        summary['permutation_test']['null_r2'] = perm['null_r2']
        print(f'  real R^2  = {perm["real_r2"]:+.4f}')
        print(f'  null mean = {perm["null_mean"]:+.4f}  std = {perm["null_std"]:.4f}')
        print(f'  p-value   = {perm["p_value"]:.4f}')

        print(f'  Feature importance ...', flush=True)
        imp = feature_importance(X_lag, y_best, args)
        summary['feature_importance'] = {
            INPUT_NAMES[i]: float(imp[i]) for i in range(len(imp))
        }
        top5 = np.argsort(imp)[::-1][:5]
        print(f'  top-5 входов: {[INPUT_NAMES[i] for i in top5]}')
        plot_feature_importance(
            imp, os.path.join(args.results_dir, 'feature_importance.png')
        )

    print(f'\n[4/4] Heatmap и сохранение ...', flush=True)
    plot_heatmap(r2_grid, args.lags, os.path.join(args.results_dir, 'r2_heatmap.png'))
    with open(os.path.join(args.results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n== ВЕРДИКТ ==')
    if best_r2 < 0.05:
        verdict = f'R^2 ~= 0 ({best_r2:+.3f}): X НЕ влияет на Y. Данные нерабочие.'
    elif best_r2 < 0.2:
        verdict = f'R^2={best_r2:+.3f}: связь слабая. Данные сомнительного качества.'
    elif best_r2 < 0.5:
        verdict = f'R^2={best_r2:+.3f}: связь заметная. Данные приемлемого качества.'
    else:
        verdict = f'R^2={best_r2:+.3f}: связь сильная. Данные хорошего качества.'
    print(verdict)
    summary['verdict'] = verdict
    with open(os.path.join(args.results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
