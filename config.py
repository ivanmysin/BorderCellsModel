"""Configuration for border cell simulation."""

# ============================================================
# Paths
# ============================================================
NEURON_PARAMS_CSV = "data/DG_CA2_Sub_CA3_CA1_EC_neuron_parameters06-30-2024_10_52_20.csv"
SYNAPSE_PARAMS_CSV = "data/DG_CA2_Sub_CA3_CA1_EC_conn_parameters06-30-2024_10_52_20.csv"
RESULTS_DIR = "results"
TRAJECTORY_HDF5 = "data/trajectory.h5"  # saved trajectory file

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
    "Input→Pyramidal":     ("CA1", "CA1 Back-Projection", "CA1", "CA1 Pyramidal"),
    "Input→Basket":        ("CA1", "CA1 Back-Projection", "CA1", "CA1 Basket"),
    "Input→Axoaxonic":     ("CA1", "CA1 Back-Projection", "CA1", "CA1 Axo-Axonic"),
}

# ========================================================
DT = 10 # 0.05 # ms
# ============================================================
# Arena & Trajectory (RatInABox)
# ============================================================
ARENA_SIZE = 0.5               # 1×1 m arena
ARENA_CM = ARENA_SIZE * 100    # 100 cm
TRAJECTORY_DT = DT * 0.001      # sec RatInABox step
SPEED_MEAN = 0.3               # mean speed m/s = 30 cm/s
THIGMOTAXIS = 0.6               # wall preference
RANDOM_SEED = 42

# ============================================================
# Training parameters
# ============================================================
SIM_DT = DT                     # 1 ms neural timestep
UP_SAMPLE_FACTOR = 10           # neural steps per RatInABox step
TRIAL_DURATION = 10.0           # 10 s per trial
N_TRIALS = 180                  # ~30 min total = 180 trials × 10 s
BATCH_SIZE = 1
N_BATCHES = N_TRIALS
LEARNING_RATE = 1e-3
GRAD_METHOD = "adjoint"        # "adjoint" | "bptt"
INTEGRATOR = "rk4"            # "euler" | "heun" | "rk4"

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
    "beta_1": 0.2,              # Hz/(cm/s), speed slope
}

# --- HD population vector generator ---
HD_POPVEC = {
    "n_hd": 18,                 # number of HD cells
    "theta_step": 20.0,         # deg, step between preferred directions
    "f_max_hd": 25.0,           # Hz, peak HD firing rate
    "kappa_hd": 3.0,            # von Mises concentration
}

# Derived
THETA_PREF = [i * HD_POPVEC["theta_step"] for i in range(HD_POPVEC["n_hd"])]

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
LOSS_WEIGHT_FR = 0.1
LOSS_WEIGHT_SPARSITY = 0.05

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

GSYN_SCALE_DIMENSIONAL = 0.5     # scale factor for gsyn_max in dimensional mode

# ============================================================
# Synapse reversal potentials (dimensional, mV)
# ============================================================
E_REV_EXC_DIM = 0.0              # AMPA reversal (mV)
E_REV_INH_DIM = -75.0            # GABA-A reversal (mV)

# ============================================================
# Tsodyks-Markram defaults (fallback if CSV lookup fails)
# ============================================================
TM_SYN_DEFAULTS = {
    "Exc→Exc": {"gsyn_max": 1.72, "tau_d": 783.2, "tau_r": 7.88, "tau_f": 13.6, "Uinc": 0.238},
    "Inh→Exc": {"gsyn_max": 6.07, "tau_d": 637.4, "tau_r": 4.41, "tau_f": 11.6, "Uinc": 0.283},
    "Inh→Inh": {"gsyn_max": 3.32, "tau_d": 635.5, "tau_r": 3.83, "tau_f": 15.1, "Uinc": 0.274},
}

E_REV_EXC = 0.0
E_REV_INH = -75.0

# ============================================================
# Trainable flags
# ============================================================
TRAIN_SYNAPSE_GMAX = True
TRAIN_SYNAPSE_U = True
TRAIN_SYNAPSE_TAU = False
TRAIN_POP_IEXT = True
TRAIN_POP_DELTA_I = False

# ============================================================
# Output
# ============================================================
SAVE_EVERY_N_TRIALS = 30
PRINT_EVERY_N_TRIALS = 5
