"""Visualization scripts for border cell simulation results."""

import os
import numpy as np
import matplotlib.pyplot as plt
import config


def set_style():
    plt.style.use('default')
    plt.rcParams.update({
        'figure.dpi': 120,
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
    })


def plot_loss_curves():
    """Plot loss and component curves over trials."""
    results_path = os.path.join(config.RESULTS_DIR, 'results.npz')
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    data = np.load(results_path, allow_pickle=True)
    loss = data['loss_history']
    mse = data['mse_history']
    fr = data['fr_history']
    sp = data['sp_history']

    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(loss, color='black', linewidth=0.8)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Trial')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_yscale('log')

    axes[0, 1].plot(mse, color='#2196F3', linewidth=0.8)
    axes[0, 1].set_title('MSE Loss')
    axes[0, 1].set_xlabel('Trial')
    axes[0, 1].set_ylabel('MSE (Hz²)')

    axes[1, 0].plot(fr, color='#4CAF50', linewidth=0.8)
    axes[1, 0].set_title('Firing Rate Regularization')
    axes[1, 0].set_xlabel('Trial')
    axes[1, 0].set_ylabel('|mean(r) - mean(r̂)|')

    axes[1, 1].plot(sp, color='#FF9800', linewidth=0.8)
    axes[1, 1].set_title('Sparsity Loss')
    axes[1, 1].set_xlabel('Trial')
    axes[1, 1].set_ylabel('1 - E[r]²/E[r²]')

    plt.suptitle('Border Cell Training Progress', fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, 'loss_curves.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_activity_over_time():
    """Plot firing rates of all populations over time."""
    sim_path = os.path.join(config.RESULTS_DIR, 'final_simulation.npz')
    if not os.path.exists(sim_path):
        print(f"Simulation file not found: {sim_path}")
        return

    data = np.load(sim_path, allow_pickle=True)
    t_seq = data['t_seq'].squeeze()  # [n_steps,] in ms
    t_sec = t_seq / 1000.0           # convert ms → s
    targets = data['targets'].squeeze()  # [n_steps, 4]

    border_names = ['Border_N', 'Border_S', 'Border_E', 'Border_W']
    other_names = ['Basket', 'Axo']

    set_style()
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    colors = {'Border_N': '#E53935', 'Border_S': '#1E88E5',
              'Border_E': '#43A047', 'Border_W': '#FB8C00'}

    for i, name in enumerate(border_names):
        ax = axes.flat[i]
        if name in data:
            fr = data[name].squeeze()
            n = min(len(fr), len(t_sec))
            ax.plot(t_sec[:n], fr[:n], color=colors[name], linewidth=0.8,
                    label='Predicted')
        if i < targets.shape[1]:
            n = min(len(targets[:, i]), len(t_sec))
            ax.plot(t_sec[:n], targets[:n, i], '--', color='gray',
                    linewidth=0.8, alpha=0.6, label='Target')
        ax.set_title(name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing rate (Hz)')
        ax.legend(fontsize=7, loc='upper right')

    for i, name in enumerate(other_names):
        ax = axes.flat[4 + i]
        if name in data:
            fr = data[name].squeeze()
            n = min(len(fr), len(t_sec))
            ax.plot(t_sec[:n], fr[:n], color='#7B1FA2', linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing rate (Hz)')

    plt.suptitle('Population Activity Over Time', fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, 'activity_over_time.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_activity_maps():
    """Plot spatial activity maps (rate vs position) for all populations."""
    sim_path = os.path.join(config.RESULTS_DIR, 'final_simulation.npz')
    if not os.path.exists(sim_path):
        print(f"Simulation file not found: {sim_path}")
        return

    data = np.load(sim_path, allow_pickle=True)
    traj_data = data['traj_arr'].item()

    x = traj_data['x'].squeeze()
    y = traj_data['y'].squeeze()

    border_names = ['Border_N', 'Border_S', 'Border_E', 'Border_W']
    all_names = border_names + ['Basket', 'Axo']
    cmaps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'winter']

    set_style()
    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))

    # Trajectory
    ax = axes[0, 0]
    ax.plot(x[:5000:5], y[:5000:5], color='gray', linewidth=0.2, alpha=0.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title('Trajectory (1×1 m)', fontsize=10)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Border populations + Basket
    for i, (name, cmap) in enumerate(zip(all_names[:4] + [all_names[4]], cmaps[:5])):
        ax = axes.flat[i + 1]
        if name in data:
            fr = data[name].squeeze()
            min_len = min(len(x), len(fr))
            # Downsample for plotting
            step = max(1, min_len // 5000)
            idx = slice(0, min_len, step)
            sc = ax.scatter(x[idx], y[idx], c=fr[idx], cmap=cmap,
                            s=2, alpha=0.7, vmin=0)
            cbar = plt.colorbar(sc, ax=ax, label='Hz', shrink=0.7)
            cbar.ax.tick_params(labelsize=7)
        ax.set_title(name, fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    # Axo-axonic separately
    ax = axes[1, 1]
    name = 'Axo'
    if name in data:
        fr = data[name].squeeze()
        min_len = min(len(x), len(fr))
        step = max(1, min_len // 5000)
        idx = slice(0, min_len, step)
        sc = ax.scatter(x[idx], y[idx], c=fr[idx], cmap='winter',
                        s=2, alpha=0.7, vmin=0)
        cbar = plt.colorbar(sc, ax=ax, label='Hz', shrink=0.7)
        cbar.ax.tick_params(labelsize=7)
    ax.set_title(name, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Turn off unused subplots
    for ax in axes.flatten()[len(all_names) + 1:]:
        ax.set_visible(False)

    plt.suptitle('Spatial Activity Maps', fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, 'activity_maps.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_all():
    """Run all plots."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plot_loss_curves()
    plot_activity_over_time()
    plot_activity_maps()


if __name__ == '__main__':
    plot_all()
