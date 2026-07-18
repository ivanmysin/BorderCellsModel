"""Configuration for border cell simulation."""

import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# Paths (project-root-relative via pathlib)
# ============================================================
NEURON_PARAMS_CSV = str(PROJECT_ROOT / "data/DG_CA2_Sub_CA3_CA1_EC_neuron_parameters06-30-2024_10_52_20.csv")
SYNAPSE_PARAMS_CSV = str(PROJECT_ROOT / "data/DG_CA2_Sub_CA3_CA1_EC_conn_parameters06-30-2024_10_52_20.csv")
RESULTS_DIR = str(PROJECT_ROOT / "results")
TRAJECTORY_HDF5 = str(PROJECT_ROOT / "data/trajectory.h5")

# ============================================================
# CSV → Population mappings
# ============================================================
NEURON_TYPE_MAP = {
    "Pyramidal": "EC LI-II Multipolar-Pyramidal",
    "Basket": "CA1 Basket",
    "Axoaxonic": "CA1 Axo-Axonic",
}

SYNAPSE_TYPE_MAP = {
    "Pyramidal→Pyramidal": ("CA1", "CA1 Pyramidal", "CA1", "CA1 Pyramidal"),
    "Pyramidal→Basket":    ("CA1", "CA1 Pyramidal", "CA1", "CA1 Basket"),
    "Pyramidal→Axoaxonic": ("CA1", "CA1 Pyramidal", "CA1", "CA1 Axo-Axonic"),
    "Basket→Pyramidal":    ("CA1", "CA1 Basket", "CA1", "CA1 Pyramidal"),
    "Basket→Basket":       ("CA1", "CA1 Basket", "CA1", "CA1 Basket"),
    "Basket→Axoaxonic":    ("CA1", "CA1 Basket", "CA1", "CA1 Axo-Axonic"),
    "Axoaxonic→Pyramidal": ("CA1", "CA1 Axo-Axonic", "CA1", "CA1 Pyramidal"),
    "Axoaxonic→Basket":    ("CA1", "CA1 Axo-Axonic", "CA1", "CA1 Basket"),
    "Axoaxonic→Axoaxonic": ("CA1", "CA1 Axo-Axonic", "CA1", "CA1 Axo-Axonic"),

    "Input→Pyramidal":     ("CA1", "CA1 Pyramidal", "CA1", "CA1 Pyramidal"),
    "Input→Basket":        ("CA1", "CA1 Pyramidal", "CA1", "CA1 Basket"),
    "Input→Axoaxonic":     ("CA1", "CA1 Pyramidal", "CA1", "CA1 Axo-Axonic"),
}

# ========================================================
DT = 0.5  # ms (neural simulation timestep)
# ============================================================
# Arena & Trajectory (RatInABox)
# ============================================================
ARENA_SIZE = 0.5               # 1×1 m arena
ARENA_CM = ARENA_SIZE * 100    # 50 cm
TRAJECTORY_DT = DT * 0.001         # ms, RatInABox timestep, 0.5 ms ≈ 200 Hz, 0.0005 ≈ 201      # sec RatInABox step

# Single-population unit mapping
POPULATION_NAME = "border_cells"
UNIT_NAMES = ["Border_N", "Border_S", "Border_E", "Border_W", "Basket", "Axo"]
UNIT_IDX = {name: i for i, name in enumerate(UNIT_NAMES)}
N_UNITS = len(UNIT_NAMES)

# Unit type mapping (for parameter lookup)
UNIT_TYPE = {
    "Border_N": "Pyramidal",
    "Border_S": "Pyramidal",
    "Border_E": "Pyramidal",
    "Border_W": "Pyramidal",
    "Basket": "Basket",
    "Axo": "Axoaxonic",
}
SPEED_MEAN = 0.3               # mean speed m/s = 30 cm/s
THIGMOTAXIS = 0.6               # wall preference
RANDOM_SEED = 42

# ============================================================
# Training parameters
# ============================================================
SIM_DT = DT                     # neural timestep
UP_SAMPLE_FACTOR = 1            # no upsampling — trajectory already at SIM_DT
TRIAL_DURATION = 2.5           # 10 s per trial
N_TRIALS = 50                  # ~30 min total = 180 trials × 10 s
BATCH_SIZE = 25

N_BATCHES = 100
LEARNING_RATE = 1e-3
N_BATCHES_PER_EPOCH = 4      # random batches sampled per epoch (dataset has ~600)
PHASE1_BATCH_SIZE = 4         # parallel stateful trajectories per sub-model
GRAD_METHOD = "adjoint"         # "bptt" | "adjoint"
INTEGRATOR = "rk4"            # "euler" | "heun" | "rk4"

# ============================================================
# Batch & epoch parameters
# ============================================================
BATCH_DURATION = 0.1            # seconds per batch
N_PARALLEL_BATCHES = 10         # how many batches to load in RAM
N_EPOCHS = 1000                 # full passes over all batches

# ============================================================
# Vectorized population
# ============================================================
N_INPUTS = 21                   # 1 d_far + 1 d_near + 1 speed + 18 HD
N_POP_UNITS = 6                 # 4 border + Basket + Axo
POPULATION_NAME = "cells"       # name for the single vectorized population

# ============================================================
# Input generator parameters
# ============================================================

# --- DistanceFar generator (active FAR from walls) ---
DISTANCE_FAR = {
    "alpha_far": 0.15,          # Hz/cm, slope (positive → far-active)
}

# --- DistanceNear generator (active NEAR walls) ---
DISTANCE_NEAR = {
    "alpha_near": 0.25,         # Hz/cm, slope magnitude (near-active)
    "d_max": 70.0,              # cm, half-diagonal → rate=0 at this distance
}

# --- Speed generator ---
SPEED_CELL = {
    "beta_0": 2.0,              # Hz, baseline firing at zero speed
    "beta_1": 0.5,              # Hz/(cm/s), speed slope
}

# --- HD population vector generator ---
HD_POPVEC = {
    "n_hd": 18,                 # number of HD cells
    "theta_step": 20.0,         # deg, step between preferred directions
    "f_max_hd": 3.0,           # Hz, peak HD firing rate
    "kappa_hd": 3.0,            # von Mises concentration
}

# Derived
THETA_PREF = [i * HD_POPVEC["theta_step"] for i in range(HD_POPVEC["n_hd"])]

# Wall direction angles (radians), used as the prior mean for HD→border gsyn_max.
# Convention: theta = atan2(vy, vx), so NORTH (vy>0) = pi/2, EAST (vx>0) = 0,
# SOUTH (vy<0) = -pi/2, WEST (vx<0) = pi.
WALL_ANGLES = {
    0:  math.pi / 2,   # Border_N
    1: -math.pi / 2,   # Border_S
    2:  0.0,           # Border_E
    3:  math.pi,       # Border_W
}
HD_SIGMA_RAD = math.radians(20.0)   # bandwidth of HD→border direction preference

# Backward-compatible aliases
ALPHA_FAR = DISTANCE_FAR["alpha_far"]
ALPHA_NEAR = DISTANCE_NEAR["alpha_near"]
D_MAX = DISTANCE_NEAR["d_max"]
BETA_0 = SPEED_CELL["beta_0"]
BETA_1 = SPEED_CELL["beta_1"]
N_HD = HD_POPVEC["n_hd"]
F_MAX_HD = HD_POPVEC["f_max_hd"]
KAPPA_HD = HD_POPVEC["kappa_hd"]

# ============================================================
# Target parameters
# ============================================================
LAMBDA_PROX = 10.0             # cm, wall proximity decay
F_MAX_BORDER = 15.0            # Hz, peak border cell firing

# ============================================================
# Loss weights
# ============================================================
LOSS_WEIGHT_MSE = 1.0
LOSS_WEIGHT_FR = 0.001
LOSS_WEIGHT_SPARSITY = 0.005
WTA_WEIGHT = 5e-2                    # decorrelation penalty weight
L2_GSYN_WEIGHT = 1e-6                # L2 penalty on gsyn_max
LOSS_WEIGHT_SHARPENING = 0.0005         # sparse border-cell activity (winner-take-all)
LOSS_WEIGHT_EI_BALANCE = 0.0005         # inhibitory activity ∝ excitatory activity

# ============================================================
# Population parameters (dimensional mode — loaded from CSV)
# ============================================================
USE_DIMENSIONAL_PARAMS = True   # True: CSV dimensional → IzhikevichMeanField

# Target dimensionless MPR parameters (neuraltide converts dimensional→dimless)
I_EXT_DIMENSIONLESS_RS = 0.2    # Regular Spiking (Pyramidal / Border)
I_EXT_DIMENSIONLESS_FS = 0.5    # Fast Spiking (Basket, Axo-axonic)
DELTA_I_DIMENSIONLESS = 0.5     # spread of input current
MPR_ALPHA_MIN = 0.5             # minimum dimensionless alpha (threshold)
MPR_TAU_POP_MIN = 50.0          # minimum tau_pop (ms), RS stable value
TAU_POP_RS = 50.0               # RS population time constant (ms)
TAU_POP_FS = 10.0               # FS population time constant (ms)
MPR_A_RS = 0.02                 # RS adaptation rate
MPR_B_RS = 0.2                  # RS adaptation coupling
MPR_WJ_RS = 0.02                # RS adaptation jump
MPR_A_FS = 0.1                  # FS adaptation rate
MPR_B_FS = 0.2                  # FS adaptation coupling
MPR_WJ_FS = 0.01                # FS adaptation jump

# When USE_DIMENSIONAL_PARAMS=True, the simulator computes I_ext_dimensional
# from the above values using the CSV-loaded K and V_T−V_rest for each cell type.
# When False, I_ext is passed directly in dimensionless form.

GSYN_SCALE_DIMENSIONAL = 0.05     # scale factor for gsyn_max in dimensional mode

# ============================================================
# Synapse reversal potentials (dimensional, mV)
# ============================================================
E_REV_EXC_DIM = 0.0              # AMPA reversal (mV)
E_REV_INH_DIM = -75.0            # GABA-A reversal (mV)

# ============================================================
# Tsodyks-Markram defaults (fallback if CSV lookup fails)
# ============================================================
TM_SYN_DEFAULTS = {
    "Exc→Exc": {"gsyn_max": 1.72, "tau_d": 5.0, "tau_r": 500.0, "tau_f": 20.6, "Uinc": 0.238},
    "Inh→Exc": {"gsyn_max": 6.07, "tau_d": 5.0, "tau_r": 500.0, "tau_f": 20.6, "Uinc": 0.283},
    "Inh→Inh": {"gsyn_max": 3.32, "tau_d": 5.0, "tau_r": 500.0, "tau_f": 20.1, "Uinc": 0.274},
}

E_REV_EXC = 0.0
E_REV_INH = -75.0

# ============================================================
# Synaptic initial state randomization & loss warmup
# ============================================================
# Initial values for R, U, A are drawn from Uniform(LO, HI) at the
# start of each batch. Adds variability across batches and helps
# break symmetry in learning. The resulting transient is masked by
# LOSS_WARMUP_STEPS so it does not contaminate the loss.
#
# Default values (LO=0.9, HI=1.0 for R and U) correspond to a fresh
# synapse ready to release. For attractor initial conditions we
# instead start the synapse in a DEPRESSED state: low R (depleted
# available pool) and low U (low release probability). This makes
# it harder for self-excitation to push a border into the "on"
# state at t=0, and so reduces the chance of getting stuck in a
# single attractor for the entire trajectory.
SYN_INIT_R_LO = 0.3
SYN_INIT_R_HI = 0.5
SYN_INIT_U_LO = 0.0
SYN_INIT_U_HI = 0.2
SYN_INIT_A_LO = 0.0
SYN_INIT_A_HI = 0.05

# Scales initial gsyn_max so I_syn starts in the active region of
# S(I_syn) even with FRpre = E * dt_dim * 0.001. Without this, the
# softplus-based dead-zone penalty has too weak a gradient to escape
# the dead zone in reasonable training time.
#   target I_syn ≈ 5  →  gsyn_max_init ≈ 5 / (A * 0.005 * sign_ei)
#   for A ≈ 1, sign_ei = +1: gsyn_max_init ≈ 1000
SYN_GSYN_INIT_SCALE = 1.0

# Number of initial steps to exclude from loss and metrics. At
# DT=0.1 ms, 500 steps = 50 ms — enough for both the synaptic
# transient (~10 steps) and the Wilson-Cowan transient (tau≈12 ms
# → ~120 steps) to settle.
LOSS_WARMUP_STEPS = 500

# ============================================================
# Synaptic dead-zone penalty (escape from Naka-Rushton S'(0)=0)
# ============================================================
# Softplus-based penalty that pushes I_syn out of the dead zone
# (I_syn <= threshold). Has non-zero gradient everywhere, including
# at I_syn = 0, so it bypasses the S'(I_syn)=0 chain.
#   penalty = softplus(-(I_syn - threshold) / tau)  per timestep
#
# Calibration (with FRpre = E * dt_dim * 0.001):
#   - I_syn ~ 0 when gsyn is at its initial value (dead zone)
#   - I_syn ~ 5 corresponds to E ~ 50 Hz (active range)
#   - threshold = 2.0 marks the boundary of usable S'(I_syn)
SYN_DEAD_ZONE_THRESHOLD = 2.0   # I_syn <= this is "dead"
SYN_DEAD_ZONE_TAU = 1.0          # softness (smaller = sharper transition)
SYN_DEAD_ZONE_WEIGHT = 0.0001    # safety net only — gsyn_init is already in active region

# ============================================================
# Trainable flags
# ============================================================
TRAIN_SYNAPSE_GMAX = True
TRAIN_SYNAPSE_U = True
TRAIN_SYNAPSE_TAU_f = True
TRAIN_SYNAPSE_TAU_r = True
TRAIN_SYNAPSE_TAU_d = True


TRAIN_POP_IEXT = True
TRAIN_POP_DELTA_I = False

# ============================================================
# Initial state ranges for BorderMeanFieldNetwork (used when
# --learnable-init-state is set in train_simple.py). Defaults reproduce
# the legacy random sampling in BorderMeanFieldNetwork.get_initial_state().
# ============================================================
BORDER_INIT_R_LO = 0.0
BORDER_INIT_R_HI = 0.1
BORDER_INIT_V_MEAN = 0.0
BORDER_INIT_V_STD = 0.01
BORDER_INIT_W_VAL = 0.0
BORDER_INIT_TM_R = 1.0
BORDER_INIT_TM_U = 0.0
BORDER_INIT_TM_A = 0.0

# ============================================================
# Wilson-Cowan initial state (used when --learnable-init-state is set)
# ============================================================
# Default LO=HI=0 preserves the "standard" zero-init behaviour. Set HI > LO
# (e.g. WC_INIT_NU_HI=10.0) to start optimisation from a random perturbation.
WC_INIT_NU_LO = 0.0
WC_INIT_NU_HI = 10.0
WC_INIT_G_LO = 0.0
WC_INIT_G_HI = 5.0
WC_INIT_DG_LO = -1.0
WC_INIT_DG_HI = 1.0

# ============================================================
# Output
# ============================================================
SAVE_EVERY_N_TRIALS = 10
PRINT_EVERY_N_TRIALS = 5

# ============================================================
# Attractor-style gsyn_max initialization (used by train_wc.py)
# ============================================================
# Role-based gsyn_max values for recurrent + input connections.
# Designed to start the network close to a point-attractor solution:
#   - 4 self-amplifying border attractors (B_X → B_X)
#   - Competition via Basket (B_X → Basket → all B_Y)
#   - Off-wall global inhibition via Axo (d_far → Axo → all Borders)
#   - All cross-border connections are 0 (WTA purely via Basket)
#
# Magnitudes calibrated to give I_syn ≈ 2-3 at 15 Hz steady state:
#   A ≈ 0.018 (for Exc→Exc, U=0.238, R=1, tau_d=5)
#   I_syn = gsyn_max × A  →  gsyn_max ≈ 130-180 for I_syn ≈ 2.5
#
# All values are FINAL gsyn_max (not multiplied by GSYN_SCALE_DIMENSIONAL).
ATTRACTOR_GSYN = {
    'SELF_EXC':       70.0,  # Border_X → Border_X (was 150; reduced to avoid bistable lock-in)
    'B_TO_BASKET':    70.0,  # Border_X → Basket
    'BASKET_TO_B':   120.0,  # Basket → Border_X
    'BASKET_SELF':    50.0,  # Basket → Basket
    'AXO_TO_B':      100.0,  # Axo → Border_X
    'AXO_SELF':       40.0,  # Axo → Axo
    'DFAR_TO_AXO':   180.0,  # d_far → Axo (primary off-wall drive)
    'DFAR_TO_BASKET': 30.0,  # d_far → Basket (weak auxiliary drive)
}

# Per-unit I_ext for WilsonCowanNetwork. Borders get a low baseline (0.3) so
# they can be SILENT when no input arrives (target rate ≈ 0 at center).
# Basket and Axo keep the standard 1.0 so they fire from rest and provide
# tonic inhibition that borders must overcome to win the WTA.
BORDER_INIT_I_EXT = 0.3
BASKET_INIT_I_EXT = 1.0
AXO_INIT_I_EXT = 1.0

# Initial firing rate of Axo at t=0. The agent always starts at the arena
# center, so d_far is at its maximum and Axo should be in its active state
# from the very first step. Setting AXO_INIT_RATE > 0 means borders are
# already inhibited at t=0 (no silent→Axo-ramps-up transient).
AXO_INIT_RATE = 30.0  # Hz
