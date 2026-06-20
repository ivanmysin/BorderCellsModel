"""Test synaptic sign in isolation.

Method: zero ALL gsyn_max and ALL input pconn. Use I_ext to drive neurons.
Only the 2 test connections (excitatory + inhibitory) are active.

Test 1: Border_N fires via I_ext → Border_E should increase (excitatory)
Test 2: Basket fires via I_ext  → Border_E should decrease (inhibitory)

Usage:
    python test_synaptic_sign.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
import train_simple
from utils.params import build_pop_params
from utils.csv_loader import get_neuron_ei, get_neuron_vr


def _dimless_e_r(pre_type, post_type):
    pre_csv = config.NEURON_TYPE_MAP.get(pre_type, pre_type)
    post_csv = config.NEURON_TYPE_MAP.get(post_type, post_type)
    E_rev = 0.0 if get_neuron_ei(pre_csv) == 'e' else -75.0
    return 1.0 + E_rev / abs(get_neuron_vr(post_csv))


def build_test_params():
    pop = build_pop_params()
    n_pre = config.N_POP_UNITS + config.N_INPUTS
    n_post = config.N_POP_UNITS

    gsyn = np.zeros((n_pre, n_post), dtype=np.float32)
    tau_d = np.full((n_pre, n_post), 500.0, dtype=np.float32)
    tau_r = np.full((n_pre, n_post), 5.0, dtype=np.float32)
    tau_f = np.full((n_pre, n_post), 10.0, dtype=np.float32)
    Uinc = np.full((n_pre, n_post), 0.2, dtype=np.float32)
    pconn = np.zeros((n_pre, n_post), dtype=np.float32)
    e_r = np.zeros((n_pre, n_post), dtype=np.float32)

    return {
        'alpha': np.asarray(pop['alpha'], dtype=np.float32),
        'a': np.asarray(pop['a'], dtype=np.float32),
        'b': np.asarray(pop['b'], dtype=np.float32),
        'w_jump': np.asarray(pop['w_jump'], dtype=np.float32),
        'tau_pop': np.asarray(pop['tau_pop'], dtype=np.float32),
        'I_ext': np.asarray(pop['I_ext'], dtype=np.float32),
        'Delta_I': np.asarray(pop['Delta_I'], dtype=np.float32),
        'gsyn_max': gsyn, 'tau_d': tau_d, 'tau_r': tau_r,
        'tau_f': tau_f, 'Uinc': Uinc, 'pconn': pconn, 'e_r': e_r,
    }


def make_input(n_steps, channel=0, rate=10.0, t_on=3000, t_off=7000):
    x = np.zeros((1, n_steps, config.N_INPUTS), dtype=np.float32)
    x[0, t_on:t_off, channel] = rate
    return x


def run_test():
    config.RANDOM_SEED = 42
    tf.random.set_seed(42)
    np.random.seed(42)

    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import RNN

    n_steps = 10000
    t_ms = np.arange(n_steps) * config.DT

    print(f"e_r excitatory  (Pyra→Pyra):  {_dimless_e_r('Pyramidal', 'Pyramidal'):.4f}")
    print(f"e_r inhibitory  (Basket→Pyra): {_dimless_e_r('Basket', 'Pyramidal'):.4f}")
    print()

    # ================================================================
    # TEST 1: Excitatory Border_N(0) → Border_E(2)
    #
    # Control: NO N→E connection. Drive Border_N via I_ext.
    #   Border_E should NOT respond (no connection to it).
    # Test: WITH N→E connection. Drive Border_N via I_ext.
    #   Border_E should increase.
    # ================================================================
    print("=" * 60)
    print("TEST 1: Excitatory connection Border_N(0) → Border_E(2)")
    print("  Drive Border_N via I_ext, Border_E has no direct input")
    print("  With N→E: Border_E should increase")
    print("  Without N→E: Border_E should stay flat")
    print("=" * 60)

    # --- Control: no N→E ---
    params_ctrl = build_test_params()
    cell_ctrl = train_simple.BorderMeanFieldNetwork(params_ctrl, dt_dim=config.DT)
    inp = Input(shape=(None, config.N_INPUTS), batch_size=1)
    rnn_ctrl = RNN(cell_ctrl, return_sequences=True, stateful=True)
    model_ctrl = Model(inp, rnn_ctrl(inp))
    rnn_ctrl.reset_states()

    i_ext_ctrl = np.zeros(6, dtype=np.float32)
    i_ext_ctrl[0] = 0.4  # drive Border_N
    cell_ctrl.I_ext.assign(i_ext_ctrl)

    x_zero = np.zeros((1, n_steps, config.N_INPUTS), dtype=np.float32)
    y_ctrl = model_ctrl.predict(x_zero, verbose=0)
    r_e_ctrl = y_ctrl[0, :, 2]

    # --- Test: with N→E ---
    params_test = build_test_params()
    EXC_ROW, EXC_COL = 0, 2
    params_test['gsyn_max'][EXC_ROW, EXC_COL] = 5.0
    params_test['pconn'][EXC_ROW, EXC_COL] = 1.0
    params_test['e_r'][EXC_ROW, EXC_COL] = _dimless_e_r('Pyramidal', 'Pyramidal')

    cell_test = train_simple.BorderMeanFieldNetwork(params_test, dt_dim=config.DT)
    inp2 = Input(shape=(None, config.N_INPUTS), batch_size=1)
    rnn_test = RNN(cell_test, return_sequences=True, stateful=True)
    model_test = Model(inp2, rnn_test(inp2))
    rnn_test.reset_states()

    i_ext_test = np.zeros(6, dtype=np.float32)
    i_ext_test[0] = 0.4  # same drive on Border_N
    cell_test.I_ext.assign(i_ext_test)

    y_test = model_test.predict(x_zero, verbose=0)
    r_n_test = y_test[0, :, 0]
    r_e_test = y_test[0, :, 2]

    # Border_N should be similar in both
    print(f"  Border_N (control) steady-state: {r_e_ctrl.mean():.4f} Hz")
    print(f"  Border_N (test)    steady-state: {r_n_test.mean():.4f} Hz")
    print(f"  Border_E (control, no N→E):  mean={r_e_ctrl.mean():.4f} Hz")
    print(f"  Border_E (test, with N→E):   mean={r_e_test.mean():.4f} Hz")

    mean_ctrl = r_e_ctrl.mean()
    mean_test = r_e_test.mean()
    pass_exc = mean_test > mean_ctrl * 1.1
    print(f"  Ratio test/ctrl: {mean_test/(mean_ctrl+1e-10):.2f}x")
    print(f"  RESULT: {'PASS' if pass_exc else 'FAIL'} "
          f"(N→E should increase Border_E)")

    # ================================================================
    # TEST 2: Inhibitory Basket(4) → Border_E(2)
    #
    # Give Border_E some I_ext so it has baseline activity.
    # Control: no Bsk→E. Border_E should stay constant.
    # Test: with Bsk→E. Drive Basket via I_ext. Border_E should drop.
    # ================================================================
    print()
    print("=" * 60)
    print("TEST 2: Inhibitory connection Basket(4) → Border_E(2)")
    print("  Drive Basket via I_ext, Border_E has baseline I_ext")
    print("  With Bsk→E: Border_E should decrease")
    print("  Without Bsk→E: Border_E should stay constant")
    print("=" * 60)

    # --- Control: no Bsk→E ---
    params_ctrl2 = build_test_params()
    cell_ctrl2 = train_simple.BorderMeanFieldNetwork(params_ctrl2, dt_dim=config.DT)
    inp3 = Input(shape=(None, config.N_INPUTS), batch_size=1)
    rnn_ctrl2 = RNN(cell_ctrl2, return_sequences=True, stateful=True)
    model_ctrl2 = Model(inp3, rnn_ctrl2(inp3))
    rnn_ctrl2.reset_states()

    i_ext_c2 = np.zeros(6, dtype=np.float32)
    i_ext_c2[2] = 0.3  # Border_E baseline
    i_ext_c2[4] = 0.4  # drive Basket
    cell_ctrl2.I_ext.assign(i_ext_c2)

    y_ctrl2 = model_ctrl2.predict(x_zero, verbose=0)
    r_e_c2 = y_ctrl2[0, :, 2]
    r_bsk_c2 = y_ctrl2[0, :, 4]

    # --- Test: with Bsk→E ---
    params_test2 = build_test_params()
    INH_ROW, INH_COL = 4, 2
    params_test2['gsyn_max'][INH_ROW, INH_COL] = 5.0
    params_test2['pconn'][INH_ROW, INH_COL] = 1.0
    params_test2['e_r'][INH_ROW, INH_COL] = _dimless_e_r('Basket', 'Pyramidal')

    cell_test2 = train_simple.BorderMeanFieldNetwork(params_test2, dt_dim=config.DT)
    inp4 = Input(shape=(None, config.N_INPUTS), batch_size=1)
    rnn_test2 = RNN(cell_test2, return_sequences=True, stateful=True)
    model_test2 = Model(inp4, rnn_test2(inp4))
    rnn_test2.reset_states()

    i_ext_t2 = np.zeros(6, dtype=np.float32)
    i_ext_t2[2] = 0.3  # Border_E baseline
    i_ext_t2[4] = 0.4  # drive Basket
    cell_test2.I_ext.assign(i_ext_t2)

    y_test2 = model_test2.predict(x_zero, verbose=0)
    r_e_t2 = y_test2[0, :, 2]
    r_bsk_t2 = y_test2[0, :, 4]

    mean_ctrl2 = r_e_c2.mean()
    mean_test2 = r_e_t2.mean()
    print(f"  Basket (control) steady-state: {r_bsk_c2.mean():.4f} Hz")
    print(f"  Basket (test)    steady-state: {r_bsk_t2.mean():.4f} Hz")
    print(f"  Border_E (control, no Bsk→E):  mean={mean_ctrl2:.4f} Hz")
    print(f"  Border_E (test, with Bsk→E):   mean={mean_test2:.4f} Hz")
    pass_inh = mean_test2 < mean_ctrl2 * 0.9
    print(f"  Ratio test/ctrl: {mean_test2/(mean_ctrl2+1e-10):.2f}x")
    print(f"  RESULT: {'PASS' if pass_inh else 'FAIL'} "
          f"(Bsk→E should suppress Border_E)")

    # ================================================================
    # PLOT
    # ================================================================
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(t_ms/1000, r_n_test, label='Border_N (presynaptic)', linewidth=1.2)
    ax.plot(t_ms/1000, r_e_test, label='Border_E (N→E ON)', linewidth=1.2)
    ax.plot(t_ms/1000, r_e_ctrl, label='Border_E (N→E OFF)', linewidth=1.2, linestyle='--')
    ax.set_title('TEST 1: Excitatory (N→E)')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Rate (Hz)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_ms/1000, r_bsk_t2, label='Basket (presynaptic)', linewidth=1.2)
    ax.plot(t_ms/1000, r_e_t2, label='Border_E (Bsk→E ON)', linewidth=1.2)
    ax.plot(t_ms/1000, r_e_c2, label='Border_E (Bsk→E OFF)', linewidth=1.2, linestyle='--')
    ax.set_title('TEST 2: Inhibitory (Basket→E)')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Rate (Hz)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    gsyn_plot = np.zeros((6, 6))
    gsyn_plot[EXC_ROW, EXC_COL] = 5.0
    gsyn_plot[INH_ROW, INH_COL] = -5.0
    im = ax.imshow(gsyn_plot, cmap='RdBu_r', vmin=-6, vmax=6, aspect='equal')
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    lbl = ['N', 'S', 'E', 'W', 'Bsk', 'Axo']
    ax.set_xticklabels(lbl, fontsize=8); ax.set_yticklabels(lbl, fontsize=8)
    ax.set_title('Test connections (signed gsyn_max)')
    plt.colorbar(im, ax=ax)
    ax.text(EXC_COL, EXC_ROW, '+5.0', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    ax.text(INH_COL, INH_ROW, '-5.0', ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    ax = axes[1, 1]
    ax.axis('off')
    summary = (
        f"SYNAPTIC SIGN TEST (isolated)\n\n"
        f"e_r excitatory  (Pyra→Pyra):   {_dimless_e_r('Pyramidal', 'Pyramidal'):.4f}\n"
        f"e_r inhibitory  (Basket→Pyra):  {_dimless_e_r('Basket', 'Pyramidal'):.4f}\n\n"
        f"1. Excitatory (N→E): {'PASS' if pass_exc else 'FAIL'}\n"
        f"   E with N→E: {mean_test:.4f} vs ctrl: {mean_ctrl:.4f}\n\n"
        f"2. Inhibitory (Bsk→E): {'PASS' if pass_inh else 'FAIL'}\n"
        f"   E with Bsk→E: {mean_test2:.4f} vs ctrl: {mean_ctrl2:.4f}\n"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    plt.suptitle('Synaptic Sign Verification (isolated connections)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(config.RESULTS_DIR, 'test_synaptic_sign.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {out_path}")

    all_pass = pass_exc and pass_inh
    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"{'='*60}")
    return all_pass


if __name__ == '__main__':
    run_test()
