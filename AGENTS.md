# AGENTS.md

Model of border cells in entorhinal cortex: Izhikevich mean-field population (6 units: 4 border + Basket + Axo) with Tsodyks-Markram synapses, fed by 21 input channels (d_far, d_near, speed, 18×HD). Built on the `neuraltide` mean-field library.

## Environment

- Python 3.12 venv at `venv/`. Activate with `source venv/bin/activate`. All scripts and tests must be run from the project root and use the venv interpreter (`venv/bin/python`, `venv/bin/pytest`).
- Installed: tensorflow 2.21, neuraltide 0.1.0, ratinabox 1.15.3, h5py, numpy, pandas, matplotlib, scipy, pytest. **scikit-learn is NOT installed** — `assess_data_quality.py` will fail to import.
- No `pyproject.toml`, no `setup.py`, no `pytest.ini`, no CI, no pre-commit, no linter/formatter. Pytest is auto-discovered from `test/`.

## Pipeline (run in this order)

1. `python generate_trajectory.py [--n-trials N] [--duration S]` → `data/trajectory.h5`. By default generates `N_TRIALS=180` trajectories of `TRIAL_DURATION=10 s` each (per `config.py`), concatenated along the time axis into one stream. Each trial starts a fresh RatInABox `Agent` (random position) so the stream is **discontinuous at trial boundaries**; the network's internal state is what carries over, not the trajectory. Pass `--n-trials 1` for the legacy single-trajectory mode.
2. `python generate_dataset.py` → `data/dataset.h5`. Reads the (longer) trajectory, precomputes 21-channel inputs and 4 wall targets, splits into `BATCH_DURATION=0.1 s` chunks. With the default 1800 s stream this yields 18000 stored batches.
3. `python train_simple.py [--dataset PATH] [--epochs N] [--lr RATE] [--seed S] [--batches-per-epoch K] [--start-batch B] [--no-reset-state] [--resume W]` → `results/checkpoints/*.weights.h5` + `results/checkpoints/loss_history.json`. Uses a **stateful Keras RNN** (`RNN(cell, return_sequences=True, stateful=True, name='border_rnn')` with the `BorderMeanFieldNetwork` cell from `train_simple.py`). Training is a manual loop over `model.train_on_batch(...)` so the cell's state (r, v, w, R, U, A) propagates across the `N_BATCHES_PER_EPOCH=4` sequential batches each epoch (≈0.4 s of trajectory per epoch). State resets at the start of each epoch by default; pass `--no-reset-state` for a continuous run. Build with `batch_size=1` (single simulation; the old `config.BATCH_SIZE=1000` is unused). Pass `--start-batch B` to fix the starting batch index, otherwise it randomises each epoch.

   **Alternative: decomposed training in two phases** (works around local-minima):

   3a. `python train_phase1.py [--phase1-dir DIR] [--epochs N] [--lr 5e-3] [--seed S] [--batches-per-epoch K]` → `results/phase1/border_{N,S,E,W}_submodel.weights.h5` + `*_vars.npz`. Builds 4 sub-models in parallel logic, each with **3 units** (one pyramidal + Basket + Axo) and **24 input channels**: 21 real (d_far, d_near, speed, HD×18) + 3 teacher channels carrying the ideal targets of the other 3 pyramids (computed as `f_max_border * exp(-d/lambda)`). Each sub-model is trained with **MSE** loss on its pyramidal output only; state propagates within the cell via the stateful RNN.

   3b. `python train_phase2.py [--phase1-dir DIR] [--epochs N] [--lr RATE] [--seed S] [--batches-per-epoch K]` → `results/phase2/phase2_epoch_*.weights.h5`. Lifts sub-model weights into a fresh 6-unit full model:

   | full-model slot | source |
   |---|---|
   | Border_X self-recurrent | sub-model X's recurrent (3, 3) diagonal |
   | Border_X → Border_Y (outgoing) | sub-model Y's teacher row for X |
   | Border_Y → Border_X (incoming) | sub-model X's teacher row for Y |
   | Border_X → Basket/Axo and reverse | sub-model X's recurrent row 0, cols 1/2 |
   | Basket/Axo self & mutual | average over the 4 sub-models |
   | real inputs → Border_X | sub-model X's input column |
   | real inputs → Basket/Axo | average over the 4 sub-models |
   | `I_ext` per Border | sub-model X's `I_ext[0]`; Basket/Axo averaged |

   The lifted full model then continues with **MSE + decorrelation penalty** loss and the same stateful-RNN training loop as `train_simple.py`.

   **Alternative: Wilson-Cowan pipeline** (`train_wc_nonpsyns.py` — do NOT modify):

   3W-a. `python train_wc_phase1.py [--epochs N] [--lr 1e-3] [--seed S] [--batches-per-epoch K] [--teacher-gsyn 1.0]` → `results/phase1_wc/border_{N,S,E,W}_submodel.weights.h5` + `*_vars.npz`. Same shape as Izhikevich phase 1 (4 × 3-unit sub-models, 24 input channels = 21 real + 3 teacher). Each sub-model uses the `WilsonCowanSubNetwork` layer from `utils/submodel_wc.py` (same dynamics as `train_wc_nonpsyns.WilsonCowanNetwork`, but with parametric `n_units`/`n_post` instead of the hard-coded 6). Loss is MSE on the pyramidal output. Teacher rows are initialised with `gsyn_max = --teacher-gsyn` (default 1.0) vs. ≈1e-4 for non-teacher rows, so the teacher signal is meaningful from step 0. State (nu, g, dg) propagates across batches via the stateful RNN.

   3W-b. `python train_wc_phase2.py [--phase1-dir DIR] [--out PATH]` → `results/phase1_wc/lifted.weights.h5` + `.npz`. Lifts 4 sub-model npz's into a fresh full WC model built via `build_model()` from the **untouched** `train_wc_nonpsyns.py`. Lifted physical values (gsyn_max, tau_1, tau_2) are converted back to theta via `inv_softplus`/`log` before assignment so the effective synapse values match exactly. Save layout matches the variables `train_wc_nonpsyns.py` uses (I_ext, theta_gsyn, theta_tau_1, theta_tau_2, tau_pop).

   3W-c. Resume full training with the lifted weights: `python train_wc_nonpsyns.py --resume results/phase1_wc/lifted.weights.h5`. `train_wc_nonpsyns.py` itself is not modified — only `model.load_weights(resume)` is called inside its existing `train()` entrypoint.1000` is unused). Pass `--start-batch B` to fix the starting batch index, otherwise it randomises each epoch.

   **Alternative: decomposed training in two phases** (works around local-minima):

   3a. `python train_phase1.py [--phase1-dir DIR] [--epochs N] [--lr 5e-3] [--seed S] [--batches-per-epoch K]` → `results/phase1/border_{N,S,E,W}_submodel.weights.h5` + `*_vars.npz`. Builds 4 sub-models in parallel logic, each with **3 units** (one pyramidal + Basket + Axo) and **24 input channels**: 21 real (d_far, d_near, speed, HD×18) + 3 teacher channels carrying the ideal targets of the other 3 pyramids (computed as `f_max_border * exp(-d/lambda)`). Each sub-model is trained with **MSE** loss on its pyramidal output only; state propagates within the cell via the stateful RNN.

   3b. `python train_phase2.py [--phase1-dir DIR] [--epochs N] [--lr RATE] [--seed S] [--batches-per-epoch K]` → `results/phase2/phase2_epoch_*.weights.h5`. Lifts sub-model weights into a fresh 6-unit full model:

   | full-model slot | source |
   |---|---|
   | Border_X self-recurrent | sub-model X's recurrent (3, 3) diagonal |
   | Border_X → Border_Y (outgoing) | sub-model Y's teacher row for X |
   | Border_Y → Border_X (incoming) | sub-model X's teacher row for Y |
   | Border_X → Basket/Axo and reverse | sub-model X's recurrent row 0, cols 1/2 |
   | Basket/Axo self & mutual | average over the 4 sub-models |
   | real inputs → Border_X | sub-model X's input column |
   | real inputs → Basket/Axo | average over the 4 sub-models |
   | `I_ext` per Border | sub-model X's `I_ext[0]`; Basket/Axo averaged |

   The lifted full model then continues with **MSE + decorrelation penalty** loss and the same stateful-RNN training loop as `train_simple.py`.
4. `python visualize_results.py` → `results/loss_curve.png`, `pred_vs_target.png`, `rate_maps.png`, `inhibitory_activity.png`.
5. `python visualize_dataset.py` → `results/dataset_preview.png`.
6. `python plot_results.py` (legacy — reads `results.npz`; **broken**, the new pipeline writes `results/training.h5` and `results/checkpoints/`, not `results.npz`).
7. `python assess_data_quality.py` (requires `pip install scikit-learn matplotlib` first).

## Stateful RNN training (the key change)

The legacy `train.py` (still present, currently the production entry point per `git log`/docs) loads precomputed batches and runs BPTT per batch in isolation — the Izhikevich/Tsodyks state is re-initialised every batch so consecutive batches are not a continuous simulation. `train_simple.py` fixes this by wrapping the `BorderMeanFieldNetwork` cell in a stateful Keras `RNN` and feeding batches with `model.train_on_batch(...)`, which preserves the cell state across calls. The dataset is now expected to be the concatenation of many short trials (so consecutive stored batches correspond to consecutive time), and the cell evolves continuously across them. At trial boundaries the trajectory itself jumps (new random agent position) but the network state keeps going. This matches the user's description in the request: "каждый новый батч продолжал предыдущий".

### Cell state clipping (mandatory)

`BorderMeanFieldNetwork.call()` clips the cell state to biologically plausible ranges after every RK4 step:

| State | Range | Reason |
|-------|-------|--------|
| `r` (rate) | [0, 200] Hz | non-negative, cap at very high rate |
| `v` (voltage) | [-10, 10] | `v_max = 10` in the cell |
| `w` (adaptation) | [-50, 50] | adaptation variable |
| `R, U, A` (synapses) | [0, 1] | Tsodyks-Markram probabilities |

This is required, not optional: the Axo unit has `tau_pop ≈ 0.36 ms` (so `dt/tau ≈ 0.28`, borderline RK4), and once training pushes `I_ext`/`gsyn_max` into a sensitive region, Axo diverges within ~10 timesteps after a state reset (pre-existing FS instability, see AGENTS.md note below). Without clipping, training crashes with NaN at epoch ~3-4 when lr=5e-3. With clipping, the same hyperparameters run indefinitely and loss decreases monotonically (verified 30+ epochs).

## FS instability note (pre-existing)

The Border cell network uses an IzhikevichMeanField with `tau_pop` loaded from the CSV (CA1 Axo-Axonic: `tau_pop = 0.36 ms`). Combined with `dt = 0.1 ms`, this gives `dt/tau ≈ 0.28` for Axo — borderline stable for RK4. After enough gradient updates, parameters drift into a region where Axo diverges. `test_network.py:63-66` documents this; tests only check Border units 0-3 for finiteness. The state clipping above prevents training crashes from this instability.

## Key files

- `config.py` — single source of truth. All paths, hyperparameters, neuron/synapse type maps, trainable flags. `GRAD_METHOD='bptt'`, `N_BATCHES_PER_EPOCH=50`.
- `utils/csv_loader.py` — loads the two `data/*.csv` files (Dori-Almog 2024 hippocampal connectome).
- `utils/params.py` — builds `[6,6]` recurrent and `[21,6]` input TsodyksMarkram parameter matrices from CSV.
- `utils/submodel.py` — builds `[3,3]` recurrent + `[24,3]` input matrices for a single Border sub-model (recurrent slice of full-model + teacher-input slice from CSV).
- `utils/submodel_wc.py` — Wilson-Cowan variant: `WilsonCowanSubNetwork` (parametric n_units/n_post, same dynamics as `train_wc_nonpsyns.WilsonCowanNetwork`) + `build_submodel_wc_params(X_idx)`. Re-exports `augment_with_teachers`, `extract_target_for` from `utils/submodel.py`.
- `utils/inputs.py` — `DistanceFar/NearGenerator`, `SpeedGenerator`, `HeadDirectionGenerator`, and the NumPy `precompute_inputs` that must stay numerically identical to them.
- `utils/dataset.py` — `prepare_batches`, `save/load_dataset_hdf5` (HDF5 layout: `dataset/batch_{i}/{t_seq, inputs, targets}`, `inputs` shape `(1, T, 21)`, `targets` shape `(1, T, 4)`).
- `utils/trajectory.py` — RatInABox wrapper + `interpolate_trajectory` (upsamples coarse `TRAJECTORY_DT=0.01s` to neural `DT=0.1ms`). Also `generate_concatenated_trajectories(gen, trial_duration, n_trials)` to chain N trials into one stream.
- `train_simple.py` — full 6-unit stateful-RNN training (from-scratch). `BorderMeanFieldNetwork` layer is parametric in `units`/`pre`/`post` so the same code builds sub-models and full models.
- `train_phase1.py` — 4 sub-models × 3 units, MSE loss, teacher-forced. Saves weights + var npz per sub-model.
- `train_phase2.py` — lifts 4 sets of sub-model vars into the full 6-unit model and trains further with MSE + WTA.
- `train_wc_phase1.py` — same shape as `train_phase1.py` but for Wilson-Cowan: 4 sub-models with 3 units, 24 inputs, MSE loss. State (nu, g, dg) propagates across batches.
- `train_wc_phase2.py` — lifts 4 WC sub-model npz's into a fresh full WC model built via `train_wc_nonpsyns.build_model()` (no modification to `train_wc_nonpsyns.py`). Saves `lifted.weights.h5` ready for `train_wc_nonpsyns.py --resume`.
- `train.py` — current vectorized pipeline; builds graph with one `graph.declare_input('inputs', n_units=21)`, then a 6-unit `IzhikevichMeanField` population, with input and recurrent `TsodyksMarkramSynapse` blocks. Training uses the local `_bptt_step` `@tf.function` (target is a tensor arg, so the trace is shared across batches). `TRAIN_*` flags in config select trainable params. Basket/Axo are padded to 6 units with target rate 0.
- `simulate.py` — legacy pipeline. The `SimulationRunner.train()` body is **commented out**; do not extend it. Uses `extra_inputs_seq` (x,y,vx,vy) fed at runtime, not precomputed inputs.
- `assess_data_quality.py` — GBT regression of targets on inputs across lags. Writes to `results/assessment/`. Supports resume via `grid.npz`.

## Critical conventions

- **Unit index order** in the 6-unit population: `[Border_N, Border_S, Border_E, Border_W, Basket, Axo]`. Targets `[B,T,4]` must be padded to `[B,T,6]` (last two are 0) for the border MSE loss.
- **Neural timestep** `config.DT = 0.1 ms`; trajectory `TRAJECTORY_DT = 0.01 s`; `UP_SAMPLE_FACTOR = 1` (trajectory is interpolated to neural dt before batching).
- **Arena is 1×1 m** = `ARENA_CM = 100 cm`. All position/distance units are cm; firing rates are Hz; time in `t_seq` is ms.
- **Input precompute must match generator formulas exactly** (d_min from `arena - y, y, arena - x, x`; HD uses von Mises with `f_max / exp(kappa)` normalization; d_near clipped to ≥ 0). See `utils/inputs.py:120` `precompute_inputs`.
- **Determinism**: set `tf.random.set_seed(config.RANDOM_SEED)`, `nt.seed_everything(config.RANDOM_SEED)`, and `np.random.seed(config.RANDOM_SEED)` together before any graph construction. Synapse `gsyn_max` matrices are randomly perturbed ±30% on each build in `utils/params.py` — different `numpy` random state ⇒ different initial network. Batch sampling per epoch also consumes the numpy random state, so per-epoch samples differ but are reproducible.
- **FS instability**: Basket and Axo units may produce NaN with random inputs at this configuration (`test_network.py:63-66` documents this; tests check only Border cells 0-3 for finiteness).
- **CSV connection mapping** lives in `config.SYNAPSE_TYPE_MAP`. Connections not found in the CSV fall back to `config.TM_SYN_DEFAULTS` (`Exc→Exc`, `Inh→Exc`, `Inh→Inh`).

## Gradient method (BPTT vs adjoint)

`train.py` uses a custom `@tf.function _bptt_step` (BPTT) rather than the `neuraltide.Trainer` adjoint path. **Do not** switch back to `grad_method='adjoint'` in the `Trainer`: the discrete adjoint backward loop in `neuraltide/training/adjoint.py:_compiled_backward_loop` runs ~30 ms/step (verified) — 30× slower than BPTT (~1 ms/step) — and would push a single epoch to ~50 hours. The analytical adjoint (`use_analytical_adjoint=True`) is **broken** in this `neuraltide` version when the graph has external inputs (`_build_analytical_index` raises `ValueError: 'inputs' is not in list`). The reason `train.py` doesn't use the `Trainer` at all is that its `@tf.function`-wrapped `_train_step` captures `self.loss_fn` (a Python object) at first trace, so reassigning `trainer.loss_fn` per batch does not actually update the targets inside the cached graph. The custom step takes the target tensor as an explicit argument, sidestepping this issue.

## Known broken / half-finished

- `train.py` does **not** export `graph_pack_inputs`, but `test/test_network.py`, `test/test_integration.py`, and `visualize_results.py` import it. These three files fail to collect (`ImportError`). The actual packing used in `train.py` is `network._graph.pack_inputs({'inputs': tf.constant(...)})` — to fix, add a thin shim `def graph_pack_inputs(network, x): return network._graph.pack_inputs({'inputs': tf.constant(x, dtype=tf.float32)})` to `train.py`.
- `plot_results.py` reads `results.npz` which `train.py` no longer writes. Use `visualize_results.py` instead.
- `assess_data_quality.py` requires `scikit-learn`; install before running.

## Testing

- `pytest test/test_dataset.py` and `test/test_generators.py` pass (30 tests). `test/test_network.py` and `test/test_integration.py` fail at collection (see above).
- `test/test_distance_far.py`, `test_distance_near.py`, `test_hd.py`, `test_speed.py` are **script-style** — they have `if __name__ == "__main__": test_xxx()` and return `bool`, so pytest warns and they are also runnable as `python test/test_xxx.py` to write plots into `results/tests/`.
- No integration CI — run tests manually. Heavy tests touch TF graph construction; expect ~10s per module.

## Docs and references

- `context/spec_border_cell_minimal.md` — model spec (inputs, parameters, design rationale, citations to Long 2025 / Kropff 2015). The `context/` directory is gitignored and contains `neuraltide` library docs (`core.md`, `populations.md`, `synapses.md`, etc.) plus runnable examples — read these before changing how the network is built.
- `README.md` is a 1-line stub. This file is the primary orientation.
