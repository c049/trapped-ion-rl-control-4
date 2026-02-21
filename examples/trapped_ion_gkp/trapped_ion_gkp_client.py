import logging
import os
import sys
import time

import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

if os.environ.get("DQ_FORCE_GPU", "0") == "1":
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

from quantum_control_rl_server.remote_env_tools import Client
from trapped_ion_gkp_sim_function import (
    trapped_ion_gkp_sim,
    trapped_ion_gkp_sim_batch,
    characteristic_grid,
    prepare_characteristic_distribution,
    characteristic_norm,
    gkp_target_fock_statistics,
)

logger = logging.getLogger("RL")
logger.propagate = False
logger.handlers = []
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

client_socket = Client()
(host, port) = (
    os.environ.get("QCRL_HOST", "127.0.0.1").strip() or "127.0.0.1",
    int(os.environ.get("QCRL_PORT", "5555")),
)
if port <= 0 or port > 65535:
    raise ValueError(f"QCRL_PORT must be in [1,65535], got {port}")
client_socket.connect((host, port))

FAST_SMOKE = os.environ.get("FAST_SMOKE", "0") == "1"

N_STEPS = int(os.environ.get("N_STEPS", "120"))
N_SEGMENTS = int(os.environ.get("N_SEGMENTS", "60"))
if N_SEGMENTS <= 0 or N_STEPS <= 0:
    raise ValueError("N_STEPS and N_SEGMENTS must both be positive.")
if N_STEPS % N_SEGMENTS != 0:
    raise ValueError(
        f"N_STEPS ({N_STEPS}) must be divisible by N_SEGMENTS ({N_SEGMENTS})."
    )
SEG_LEN = N_STEPS // N_SEGMENTS
_t_step_env = os.environ.get("T_STEP", "").strip()
if _t_step_env:
    T_STEP = float(_t_step_env)
    if not np.isfinite(T_STEP) or T_STEP <= 0.0:
        raise ValueError(f"T_STEP must be > 0 and finite, got {T_STEP}")
    TOTAL_DURATION = float(N_STEPS) * T_STEP
else:
    TOTAL_DURATION = float(os.environ.get("TOTAL_DURATION", "1200.0"))
    if not np.isfinite(TOTAL_DURATION) or TOTAL_DURATION <= 0.0:
        raise ValueError(
            f"TOTAL_DURATION must be > 0 and finite, got {TOTAL_DURATION}"
        )
    T_STEP = TOTAL_DURATION / float(N_STEPS)
SAMPLE_EXTENT = float(os.environ.get("SAMPLE_EXTENT", "4.0"))
PLOT_EXTENT = float(os.environ.get("PLOT_EXTENT", "6.0"))
N_BOSON = int(os.environ.get("N_BOSON", "40"))
if N_BOSON <= 2:
    raise ValueError(f"N_BOSON must be > 2, got {N_BOSON}")
GKP_DELTA = float(os.environ.get("GKP_DELTA", "0.301"))
GKP_LOGICAL = os.environ.get("GKP_LOGICAL", "0")
_gkp_delta_curr_env = os.environ.get("GKP_DELTA_CURRICULUM", "").strip()
if _gkp_delta_curr_env:
    GKP_DELTA_CURRICULUM = [
        float(v.strip()) for v in _gkp_delta_curr_env.split(",") if v.strip()
    ]
else:
    GKP_DELTA_CURRICULUM = [GKP_DELTA]
_gkp_delta_curr_epochs_env = os.environ.get("GKP_DELTA_CURRICULUM_EPOCHS", "").strip()
if _gkp_delta_curr_epochs_env:
    GKP_DELTA_CURRICULUM_EPOCHS = [
        int(v.strip()) for v in _gkp_delta_curr_epochs_env.split(",") if v.strip()
    ]
else:
    GKP_DELTA_CURRICULUM_EPOCHS = []
if len(GKP_DELTA_CURRICULUM) > 1:
    if len(GKP_DELTA_CURRICULUM_EPOCHS) != len(GKP_DELTA_CURRICULUM) - 1:
        raise ValueError(
            "GKP_DELTA_CURRICULUM_EPOCHS must have len(GKP_DELTA_CURRICULUM)-1 entries."
        )
else:
    GKP_DELTA_CURRICULUM_EPOCHS = []
for delta_value in GKP_DELTA_CURRICULUM:
    if not np.isfinite(delta_value) or delta_value <= 0.0:
        raise ValueError(f"Each GKP delta must be positive and finite, got {delta_value}")
for boundary in GKP_DELTA_CURRICULUM_EPOCHS:
    if boundary < 0:
        raise ValueError("GKP_DELTA_CURRICULUM_EPOCHS entries must be >= 0.")
for i in range(1, len(GKP_DELTA_CURRICULUM_EPOCHS)):
    if GKP_DELTA_CURRICULUM_EPOCHS[i] <= GKP_DELTA_CURRICULUM_EPOCHS[i - 1]:
        raise ValueError("GKP_DELTA_CURRICULUM_EPOCHS must be strictly increasing.")
_gkp_squeeze_env = os.environ.get("GKP_SQUEEZE_R", "").strip()
GKP_SQUEEZE_R = None if _gkp_squeeze_env == "" else float(_gkp_squeeze_env)
_gkp_kappa_env = os.environ.get("GKP_ENVELOPE_KAPPA", "").strip()
GKP_ENVELOPE_KAPPA = None if _gkp_kappa_env == "" else float(_gkp_kappa_env)
GKP_LATTICE_TRUNC = int(os.environ.get("GKP_LATTICE_TRUNC", "4"))
GKP_TAIL_WARN = float(os.environ.get("GKP_TARGET_TAIL_WARN", "1.0e-3"))
GKP_TAIL_ERROR = float(os.environ.get("GKP_TARGET_TAIL_ERROR", "5.0e-3"))
ALLOW_LOW_N_BOSON = os.environ.get("ALLOW_LOW_N_BOSON", "0") == "1"

TRAIN_POINTS_STAGE1 = int(os.environ.get("TRAIN_POINTS_STAGE1", "120"))
TRAIN_POINTS_STAGE2 = int(os.environ.get("TRAIN_POINTS_STAGE2", "240"))
TRAIN_POINTS_STAGE3 = int(os.environ.get("TRAIN_POINTS_STAGE3", "960"))
TRAIN_STAGE1_EPOCHS = int(os.environ.get("TRAIN_STAGE1_EPOCHS", "0"))
TRAIN_STAGE2_EPOCHS = int(os.environ.get("TRAIN_STAGE2_EPOCHS", "180"))

CHAR_GRID_SIZE = int(os.environ.get("CHAR_GRID_SIZE", "61"))
FINAL_GRID_SIZE = int(os.environ.get("FINAL_GRID_SIZE", "61"))
PLOT_GRID_SIZE = int(os.environ.get("PLOT_GRID_SIZE", "121"))
if CHAR_GRID_SIZE < 3 or FINAL_GRID_SIZE < 3 or PLOT_GRID_SIZE < 3:
    raise ValueError("CHAR_GRID_SIZE, FINAL_GRID_SIZE and PLOT_GRID_SIZE must be >= 3")

if FAST_SMOKE:
    N_BOSON = 12
    TRAIN_POINTS_STAGE1 = 20
    TRAIN_POINTS_STAGE2 = 40
    TRAIN_POINTS_STAGE3 = 60
    TRAIN_STAGE1_EPOCHS = 2
    TRAIN_STAGE2_EPOCHS = 4
    CHAR_GRID_SIZE = 21
    FINAL_GRID_SIZE = 31
    PLOT_GRID_SIZE = 61

SMOOTH_LAMBDA = 0.0
SMOOTH_PHI_WEIGHT = 1.0
SMOOTH_AMP_WEIGHT = 0.2
REWARD_SCALE = 1.0
REWARD_CLIP = None
CHAR_REWARD_OBJECTIVE = os.environ.get("CHAR_REWARD_OBJECTIVE", "overlap_real").lower()
CHAR_REWARD_OBJECTIVE_STAGE2 = os.environ.get("CHAR_REWARD_OBJECTIVE_STAGE2", "").strip().lower()
CHAR_REWARD_SWITCH_EPOCH = int(os.environ.get("CHAR_REWARD_SWITCH_EPOCH", "-1"))
CHAR_REWARD_SWITCH_AUTO_CLAMP = (
    os.environ.get("CHAR_REWARD_SWITCH_AUTO_CLAMP", "1") == "1"
)
CHAR_REWARD_SWITCH_AUTO_FRACTION = float(
    os.environ.get("CHAR_REWARD_SWITCH_AUTO_FRACTION", "0.60")
)
CHAR_REWARD_SWITCH_MIN_BEST_EVAL = float(
    os.environ.get("CHAR_REWARD_SWITCH_MIN_BEST_EVAL", "-1.0")
)
CHAR_REWARD_STAGE2_PATIENCE_EVAL = int(
    os.environ.get("CHAR_REWARD_STAGE2_PATIENCE_EVAL", "12")
)
CHAR_REWARD_STAGE2_MIN_GAIN = float(
    os.environ.get("CHAR_REWARD_STAGE2_MIN_GAIN", "0.004")
)
CHAR_REWARD_STAGE2_ALLOW_REVERT = (
    os.environ.get("CHAR_REWARD_STAGE2_ALLOW_REVERT", "1") == "1"
)
CHAR_REWARD_SWITCH_FORCE_AFTER_EVAL_PATIENCE = int(
    os.environ.get("CHAR_REWARD_SWITCH_FORCE_AFTER_EVAL_PATIENCE", "-1")
)
if CHAR_REWARD_OBJECTIVE_STAGE2 == "":
    CHAR_REWARD_OBJECTIVE_STAGE2 = CHAR_REWARD_OBJECTIVE
_valid_char_objectives = {"overlap_real", "overlap_abs", "nmse", "nmse_exp"}
if CHAR_REWARD_OBJECTIVE not in _valid_char_objectives:
    raise ValueError(
        f"Unsupported CHAR_REWARD_OBJECTIVE={CHAR_REWARD_OBJECTIVE}, "
        f"expected one of {sorted(_valid_char_objectives)}"
    )
if CHAR_REWARD_OBJECTIVE_STAGE2 not in _valid_char_objectives:
    raise ValueError(
        f"Unsupported CHAR_REWARD_OBJECTIVE_STAGE2={CHAR_REWARD_OBJECTIVE_STAGE2}, "
        f"expected one of {sorted(_valid_char_objectives)}"
    )
if CHAR_REWARD_SWITCH_EPOCH < -1:
    raise ValueError(
        f"CHAR_REWARD_SWITCH_EPOCH must be >= -1, got {CHAR_REWARD_SWITCH_EPOCH}"
    )
if CHAR_REWARD_STAGE2_PATIENCE_EVAL < 0:
    raise ValueError(
        f"CHAR_REWARD_STAGE2_PATIENCE_EVAL must be >= 0, got {CHAR_REWARD_STAGE2_PATIENCE_EVAL}"
    )
if CHAR_REWARD_SWITCH_FORCE_AFTER_EVAL_PATIENCE < -1:
    raise ValueError(
        "CHAR_REWARD_SWITCH_FORCE_AFTER_EVAL_PATIENCE must be >= -1, "
        f"got {CHAR_REWARD_SWITCH_FORCE_AFTER_EVAL_PATIENCE}"
    )
if not np.isfinite(CHAR_REWARD_STAGE2_MIN_GAIN) or CHAR_REWARD_STAGE2_MIN_GAIN < 0.0:
    raise ValueError(
        f"CHAR_REWARD_STAGE2_MIN_GAIN must be finite and >= 0, got {CHAR_REWARD_STAGE2_MIN_GAIN}"
    )
EVAL_INTERVAL_HINT = int(os.environ.get("EVAL_INTERVAL", "10"))
if EVAL_INTERVAL_HINT <= 0:
    raise ValueError(f"EVAL_INTERVAL must be > 0, got {EVAL_INTERVAL_HINT}")
NUM_EPOCHS_HINT = int(os.environ.get("NUM_EPOCHS", "2000"))
if NUM_EPOCHS_HINT <= 0:
    raise ValueError(f"NUM_EPOCHS must be > 0, got {NUM_EPOCHS_HINT}")
if (
    not np.isfinite(CHAR_REWARD_SWITCH_AUTO_FRACTION)
    or CHAR_REWARD_SWITCH_AUTO_FRACTION <= 0.0
    or CHAR_REWARD_SWITCH_AUTO_FRACTION >= 1.0
):
    raise ValueError(
        "CHAR_REWARD_SWITCH_AUTO_FRACTION must be in (0, 1), "
        f"got {CHAR_REWARD_SWITCH_AUTO_FRACTION}"
    )

CHAR_REWARD_SWITCH_EPOCH_EFFECTIVE = CHAR_REWARD_SWITCH_EPOCH
CHAR_REWARD_SWITCH_AUTOCONFIG_NOTE = ""
if (
    CHAR_REWARD_OBJECTIVE_STAGE2 != CHAR_REWARD_OBJECTIVE
    and CHAR_REWARD_SWITCH_EPOCH_EFFECTIVE >= 0
    and CHAR_REWARD_SWITCH_EPOCH_EFFECTIVE >= NUM_EPOCHS_HINT
    and CHAR_REWARD_SWITCH_AUTO_CLAMP
):
    fallback_epoch = int(np.floor(CHAR_REWARD_SWITCH_AUTO_FRACTION * NUM_EPOCHS_HINT))
    fallback_epoch = int(np.clip(fallback_epoch, 1, max(1, NUM_EPOCHS_HINT - 1)))
    CHAR_REWARD_SWITCH_AUTOCONFIG_NOTE = (
        "Auto-clamped stage2 switch epoch from "
        f"{CHAR_REWARD_SWITCH_EPOCH_EFFECTIVE} to {fallback_epoch} "
        f"(NUM_EPOCHS={NUM_EPOCHS_HINT}, fraction={CHAR_REWARD_SWITCH_AUTO_FRACTION:.2f})"
    )
    CHAR_REWARD_SWITCH_EPOCH_EFFECTIVE = fallback_epoch

N_SHOTS_TRAIN = 0
N_SHOTS_EVAL = 0

ACTION_NOISE_PHI = 0.0
ACTION_NOISE_AMP = 0.0

CHAR_START_MODE = os.environ.get("CHAR_START_MODE", "radial_topk").lower()
CHAR_RADIAL_EXP = float(os.environ.get("CHAR_RADIAL_EXP", "1.0"))
CHAR_ALPHA_SCALE = float(os.environ.get("CHAR_ALPHA_SCALE", "1.0"))
_alpha_scales_env = os.environ.get("CHAR_ALPHA_SCALES", "").strip()
if _alpha_scales_env:
    CHAR_ALPHA_SCALES = [float(v.strip()) for v in _alpha_scales_env.split(",") if v.strip()]
else:
    CHAR_ALPHA_SCALES = [CHAR_ALPHA_SCALE]
CHAR_SAMPLER_MODE = os.environ.get("CHAR_SAMPLER_MODE", "radial_stratified").lower()
CHAR_RADIAL_BINS = int(os.environ.get("CHAR_RADIAL_BINS", "8"))
CHAR_IMPORTANCE_POWER = float(os.environ.get("CHAR_IMPORTANCE_POWER", "2.0"))
GKP_LATTICE_MIX = float(os.environ.get("GKP_LATTICE_MIX", "0.35"))
GKP_LATTICE_MIX_START = float(
    os.environ.get("GKP_LATTICE_MIX_START", f"{GKP_LATTICE_MIX}")
)
GKP_LATTICE_MIX_END = float(
    os.environ.get("GKP_LATTICE_MIX_END", f"{GKP_LATTICE_MIX}")
)
GKP_LATTICE_MIX_ANNEAL_EPOCHS = int(
    os.environ.get("GKP_LATTICE_MIX_ANNEAL_EPOCHS", "0")
)
GKP_LATTICE_TOPK_BOOST = float(os.environ.get("GKP_LATTICE_TOPK_BOOST", "1.5"))
GKP_LATTICE_ORDER = int(os.environ.get("GKP_LATTICE_ORDER", "3"))
if (
    not np.isfinite(GKP_LATTICE_MIX_START)
    or GKP_LATTICE_MIX_START < 0.0
    or GKP_LATTICE_MIX_START > 1.0
):
    raise ValueError(
        f"GKP_LATTICE_MIX_START must be in [0, 1], got {GKP_LATTICE_MIX_START}"
    )
if (
    not np.isfinite(GKP_LATTICE_MIX_END)
    or GKP_LATTICE_MIX_END < 0.0
    or GKP_LATTICE_MIX_END > 1.0
):
    raise ValueError(
        f"GKP_LATTICE_MIX_END must be in [0, 1], got {GKP_LATTICE_MIX_END}"
    )
if GKP_LATTICE_MIX_ANNEAL_EPOCHS < 0:
    raise ValueError(
        f"GKP_LATTICE_MIX_ANNEAL_EPOCHS must be >= 0, got {GKP_LATTICE_MIX_ANNEAL_EPOCHS}"
    )
if not np.isfinite(GKP_LATTICE_TOPK_BOOST) or GKP_LATTICE_TOPK_BOOST < 0.0:
    raise ValueError(
        f"GKP_LATTICE_TOPK_BOOST must be >= 0 and finite, got {GKP_LATTICE_TOPK_BOOST}"
    )
if GKP_LATTICE_ORDER < 0:
    raise ValueError(f"GKP_LATTICE_ORDER must be >= 0, got {GKP_LATTICE_ORDER}")
if not np.isfinite(CHAR_IMPORTANCE_POWER) or CHAR_IMPORTANCE_POWER <= 0.0:
    raise ValueError(
        f"CHAR_IMPORTANCE_POWER must be > 0 and finite, got {CHAR_IMPORTANCE_POWER}"
    )
_gkp_phase_env = os.environ.get("GKP_REL_PHASE", "").strip()
GKP_REL_PHASE = None if _gkp_phase_env == "" else float(_gkp_phase_env)
PHASE_CLIP = float(os.environ.get("PHASE_CLIP", str(np.pi)))
USE_PERIODIC_PHASE_PROJECTION = (
    os.environ.get("USE_PERIODIC_PHASE_PROJECTION", "1") == "1"
)
_PHASE_WRAP_ENABLED = USE_PERIODIC_PHASE_PROJECTION and np.isclose(
    PHASE_CLIP, np.pi, atol=1.0e-9
)
AMP_MIN = float(os.environ.get("AMP_MIN", "0.0"))
AMP_MAX = float(os.environ.get("AMP_MAX", "2.0"))

CHAR_UNIFORM_MIX = float(os.environ.get("CHAR_UNIFORM_MIX", "0.5"))
FINAL_REFINE_SAMPLES = int(os.environ.get("FINAL_REFINE_SAMPLES", "512"))
FINAL_REFINE_SCALE = float(os.environ.get("FINAL_REFINE_SCALE", "1.0"))
FINAL_REFINE_SEED = int(os.environ.get("FINAL_REFINE_SEED", "1234"))
FINAL_REFINE_ROUNDS = int(os.environ.get("FINAL_REFINE_ROUNDS", "8"))
FINAL_REFINE_TOPK = int(os.environ.get("FINAL_REFINE_TOPK", "24"))
FINAL_REFINE_TOP_EVAL_CENTERS = int(
    os.environ.get("FINAL_REFINE_TOP_EVAL_CENTERS", "3")
)
FINAL_REFINE_DECAY = float(os.environ.get("FINAL_REFINE_DECAY", "0.6"))
FINAL_REFINE_MIN_SIGMA = float(os.environ.get("FINAL_REFINE_MIN_SIGMA", "0.05"))
FINAL_REFINE_ENABLE_AMP = os.environ.get("FINAL_REFINE_ENABLE_AMP", "0") == "1"
FINAL_REFINE_MIN_SIGMA_AMP = float(
    os.environ.get("FINAL_REFINE_MIN_SIGMA_AMP", "0.02")
)
FINAL_REFINE_INIT_SIGMA_AMP = float(
    os.environ.get("FINAL_REFINE_INIT_SIGMA_AMP", "0.15")
)
FINAL_REFINE_AMP_START_ROUND = int(
    os.environ.get("FINAL_REFINE_AMP_START_ROUND", "4")
)
if FINAL_REFINE_AMP_START_ROUND < 0:
    raise ValueError(
        f"FINAL_REFINE_AMP_START_ROUND must be >= 0, got {FINAL_REFINE_AMP_START_ROUND}"
    )
FINAL_REFINE_USE_LOC_CENTER = os.environ.get("FINAL_REFINE_USE_LOC_CENTER", "1") == "1"
FINAL_REFINE_USE_TRAIN_CENTER = os.environ.get("FINAL_REFINE_USE_TRAIN_CENTER", "1") == "1"
INIT_PULSES_NPZ = os.environ.get("INIT_PULSES_NPZ", "").strip()
INIT_PULSE_BLEND = float(os.environ.get("INIT_PULSE_BLEND", "1.0"))
FINAL_REFINE_USE_WARMSTART_CENTER = (
    os.environ.get("FINAL_REFINE_USE_WARMSTART_CENTER", "1") == "1"
)
FINAL_REFINE_USE_CHECKPOINT_CENTER = (
    os.environ.get("FINAL_REFINE_USE_CHECKPOINT_CENTER", "1") == "1"
)
FINAL_REFINE_CHECKPOINT_PULSES = os.environ.get(
    "FINAL_REFINE_CHECKPOINT_PULSES", "checkpoint/final_pulses_best.npz"
).strip()
FINAL_REFINE_CHECKPOINT_BLEND = float(
    os.environ.get("FINAL_REFINE_CHECKPOINT_BLEND", "1.0")
)
FINAL_REFINE_CENTER_SEED_STRIDE = int(
    os.environ.get("FINAL_REFINE_CENTER_SEED_STRIDE", "1000003")
)
FINAL_REFINE_FULL_STEPS = os.environ.get("FINAL_REFINE_FULL_STEPS", "0") == "1"
FINAL_REFINE_FULL_SAMPLES = int(os.environ.get("FINAL_REFINE_FULL_SAMPLES", "2048"))
FINAL_REFINE_FULL_ROUNDS = int(os.environ.get("FINAL_REFINE_FULL_ROUNDS", "6"))
FINAL_REFINE_FULL_TOPK = int(os.environ.get("FINAL_REFINE_FULL_TOPK", "64"))
FINAL_REFINE_FULL_SCALE = float(os.environ.get("FINAL_REFINE_FULL_SCALE", "0.6"))
FINAL_REFINE_FULL_DECAY = float(os.environ.get("FINAL_REFINE_FULL_DECAY", "0.6"))
FINAL_REFINE_FULL_CANDIDATES = int(
    os.environ.get("FINAL_REFINE_FULL_CANDIDATES", "1")
)
FINAL_REFINE_FULL_MIN_SIGMA = float(
    os.environ.get("FINAL_REFINE_FULL_MIN_SIGMA", "0.003")
)
FINAL_REFINE_FULL_SIGMA_FACTOR = float(
    os.environ.get("FINAL_REFINE_FULL_SIGMA_FACTOR", "0.5")
)
FINAL_REFINE_FULL_ENABLE_AMP = (
    os.environ.get("FINAL_REFINE_FULL_ENABLE_AMP", "0") == "1"
)
FINAL_REFINE_FULL_MIN_SIGMA_AMP = float(
    os.environ.get("FINAL_REFINE_FULL_MIN_SIGMA_AMP", "0.0015")
)
FINAL_REFINE_FULL_SIGMA_FACTOR_AMP = float(
    os.environ.get(
        "FINAL_REFINE_FULL_SIGMA_FACTOR_AMP",
        str(FINAL_REFINE_FULL_SIGMA_FACTOR),
    )
)
FINAL_REFINE_DIVERSE_CENTERS = (
    os.environ.get("FINAL_REFINE_DIVERSE_CENTERS", "1") == "1"
)
FINAL_REFINE_DIVERSITY_PHASE_RMS = float(
    os.environ.get("FINAL_REFINE_DIVERSITY_PHASE_RMS", "0.08")
)
FINAL_REFINE_DIVERSITY_AMP_RMS = float(
    os.environ.get("FINAL_REFINE_DIVERSITY_AMP_RMS", "0.008")
)
FINAL_REFINE_SEG_UPDATE_MODE = os.environ.get(
    "FINAL_REFINE_SEG_UPDATE_MODE", "best"
).strip().lower()
FINAL_REFINE_FULL_UPDATE_MODE = os.environ.get(
    "FINAL_REFINE_FULL_UPDATE_MODE", "best"
).strip().lower()
FINAL_REFINE_SEG_ELITE_FRACTION = float(
    os.environ.get("FINAL_REFINE_SEG_ELITE_FRACTION", "0.25")
)
FINAL_REFINE_FULL_ELITE_FRACTION = float(
    os.environ.get("FINAL_REFINE_FULL_ELITE_FRACTION", "0.15")
)
FINAL_REFINE_ELITE_WEIGHT_POWER = float(
    os.environ.get("FINAL_REFINE_ELITE_WEIGHT_POWER", "2.0")
)
FINAL_POLISH_ENABLE = os.environ.get("FINAL_POLISH_ENABLE", "0") == "1"
FINAL_POLISH_ITERS = int(os.environ.get("FINAL_POLISH_ITERS", "120"))
FINAL_POLISH_INCLUDE_AMP = os.environ.get("FINAL_POLISH_INCLUDE_AMP", "0") == "1"
FINAL_POLISH_TOP_CANDIDATES = int(
    os.environ.get("FINAL_POLISH_TOP_CANDIDATES", "0")
)
FINAL_POLISH_PHASE_A0 = float(os.environ.get("FINAL_POLISH_PHASE_A0", "0.03"))
FINAL_POLISH_PHASE_C0 = float(os.environ.get("FINAL_POLISH_PHASE_C0", "0.01"))
FINAL_POLISH_AMP_A0 = float(os.environ.get("FINAL_POLISH_AMP_A0", "0.01"))
FINAL_POLISH_AMP_C0 = float(os.environ.get("FINAL_POLISH_AMP_C0", "0.003"))
FINAL_POLISH_ALPHA = float(os.environ.get("FINAL_POLISH_ALPHA", "0.602"))
FINAL_POLISH_GAMMA = float(os.environ.get("FINAL_POLISH_GAMMA", "0.101"))
FINAL_POLISH_PATIENCE = int(os.environ.get("FINAL_POLISH_PATIENCE", "40"))
FINAL_POLISH_LOG_INTERVAL = int(os.environ.get("FINAL_POLISH_LOG_INTERVAL", "10"))
FINAL_POLISH_ACCEPT_MODE = os.environ.get("FINAL_POLISH_ACCEPT_MODE", "greedy").strip().lower()
FINAL_POLISH_SCOUT_ITERS = int(os.environ.get("FINAL_POLISH_SCOUT_ITERS", "0"))
FINAL_POLISH_SCOUT_PATIENCE = int(
    os.environ.get("FINAL_POLISH_SCOUT_PATIENCE", "60")
)
FINAL_POLISH_FINAL_ITERS = int(os.environ.get("FINAL_POLISH_FINAL_ITERS", "0"))
FINAL_POLISH_FINAL_TOP_CENTERS = int(
    os.environ.get("FINAL_POLISH_FINAL_TOP_CENTERS", "1")
)
FINAL_POLISH_FINAL_RESTARTS = int(
    os.environ.get("FINAL_POLISH_FINAL_RESTARTS", "1")
)
FINAL_POLISH_RESTART_PHASE_JITTER = float(
    os.environ.get("FINAL_POLISH_RESTART_PHASE_JITTER", "0.0")
)
FINAL_POLISH_RESTART_AMP_JITTER = float(
    os.environ.get("FINAL_POLISH_RESTART_AMP_JITTER", "0.0")
)
FINAL_POLISH_RESTART_DECAY = float(
    os.environ.get("FINAL_POLISH_RESTART_DECAY", "0.7")
)
FINAL_POLISH_AUTO_EXTEND = os.environ.get("FINAL_POLISH_AUTO_EXTEND", "0") == "1"
FINAL_POLISH_EXTEND_BLOCK = int(os.environ.get("FINAL_POLISH_EXTEND_BLOCK", "200"))
FINAL_POLISH_MAX_ITERS = int(
    os.environ.get("FINAL_POLISH_MAX_ITERS", str(FINAL_POLISH_ITERS))
)
FINAL_POLISH_EXTEND_MIN_GAIN = float(
    os.environ.get("FINAL_POLISH_EXTEND_MIN_GAIN", "5.0e-5")
)
FINAL_COORD_POLISH_ENABLE = os.environ.get("FINAL_COORD_POLISH_ENABLE", "0") == "1"
FINAL_COORD_POLISH_INCLUDE_AMP = (
    os.environ.get("FINAL_COORD_POLISH_INCLUDE_AMP", "1") == "1"
)
FINAL_COORD_POLISH_SWEEPS = int(os.environ.get("FINAL_COORD_POLISH_SWEEPS", "2"))
FINAL_COORD_POLISH_MAX_MOVES = int(
    os.environ.get("FINAL_COORD_POLISH_MAX_MOVES", "24")
)
FINAL_COORD_POLISH_BATCH_SIZE = int(
    os.environ.get("FINAL_COORD_POLISH_BATCH_SIZE", "192")
)
FINAL_COORD_POLISH_PHASE_STEP = float(
    os.environ.get("FINAL_COORD_POLISH_PHASE_STEP", "0.006")
)
FINAL_COORD_POLISH_PHASE_DECAY = float(
    os.environ.get("FINAL_COORD_POLISH_PHASE_DECAY", "0.6")
)
FINAL_COORD_POLISH_PHASE_MIN_STEP = float(
    os.environ.get("FINAL_COORD_POLISH_PHASE_MIN_STEP", "4.0e-4")
)
FINAL_COORD_POLISH_AMP_STEP = float(
    os.environ.get("FINAL_COORD_POLISH_AMP_STEP", "0.0015")
)
FINAL_COORD_POLISH_AMP_DECAY = float(
    os.environ.get("FINAL_COORD_POLISH_AMP_DECAY", "0.6")
)
FINAL_COORD_POLISH_AMP_MIN_STEP = float(
    os.environ.get("FINAL_COORD_POLISH_AMP_MIN_STEP", "2.0e-4")
)
FINAL_COORD_POLISH_MIN_GAIN = float(
    os.environ.get("FINAL_COORD_POLISH_MIN_GAIN", "1.0e-6")
)
FINAL_COORD_POLISH_TOP_CANDIDATES = int(
    os.environ.get("FINAL_COORD_POLISH_TOP_CANDIDATES", "1")
)
FINAL_COORD_POLISH_EMPTY_SWEEP_PATIENCE = int(
    os.environ.get("FINAL_COORD_POLISH_EMPTY_SWEEP_PATIENCE", "2")
)
FINAL_COORD_POLISH_RESERVE_SEC = float(
    os.environ.get("FINAL_COORD_POLISH_RESERVE_SEC", "0.0")
)
POST_RL_BUDGET_SEC = float(os.environ.get("POST_RL_BUDGET_SEC", "0"))
POST_RL_STOP_BUFFER_SEC = float(os.environ.get("POST_RL_STOP_BUFFER_SEC", "90"))
POST_RL_MIN_STAGE_SEC = float(os.environ.get("POST_RL_MIN_STAGE_SEC", "120"))
CHAR_USE_FIXED_REWARD_NORM = (
    os.environ.get("CHAR_USE_FIXED_REWARD_NORM", "1") == "1"
)
_train_fid_default_start = (
    int(GKP_DELTA_CURRICULUM_EPOCHS[-1])
    if GKP_DELTA_CURRICULUM_EPOCHS
    else int(TRAIN_STAGE2_EPOCHS)
)
TRAIN_FID_SCREEN_ENABLE = os.environ.get("TRAIN_FID_SCREEN_ENABLE", "1") == "1"
TRAIN_FID_SCREEN_START_EPOCH = int(
    os.environ.get("TRAIN_FID_SCREEN_START_EPOCH", str(_train_fid_default_start))
)
TRAIN_FID_SCREEN_INTERVAL = int(os.environ.get("TRAIN_FID_SCREEN_INTERVAL", "5"))
TRAIN_FID_SCREEN_TOPK = int(os.environ.get("TRAIN_FID_SCREEN_TOPK", "8"))
TRAIN_FID_SCREEN_RECORDS = int(os.environ.get("TRAIN_FID_SCREEN_RECORDS", "6"))
FINAL_REFINE_USE_TRAIN_FID_CENTER = (
    os.environ.get("FINAL_REFINE_USE_TRAIN_FID_CENTER", "1") == "1"
)
if FINAL_REFINE_FULL_SAMPLES < 0:
    raise ValueError(
        f"FINAL_REFINE_FULL_SAMPLES must be >= 0, got {FINAL_REFINE_FULL_SAMPLES}"
    )
if FINAL_REFINE_FULL_ROUNDS <= 0:
    raise ValueError(
        f"FINAL_REFINE_FULL_ROUNDS must be > 0, got {FINAL_REFINE_FULL_ROUNDS}"
    )
if FINAL_REFINE_FULL_TOPK <= 0:
    raise ValueError(
        f"FINAL_REFINE_FULL_TOPK must be > 0, got {FINAL_REFINE_FULL_TOPK}"
    )
if FINAL_REFINE_FULL_CANDIDATES <= 0:
    raise ValueError(
        f"FINAL_REFINE_FULL_CANDIDATES must be > 0, got {FINAL_REFINE_FULL_CANDIDATES}"
    )
if FINAL_REFINE_FULL_MIN_SIGMA <= 0.0:
    raise ValueError(
        f"FINAL_REFINE_FULL_MIN_SIGMA must be > 0, got {FINAL_REFINE_FULL_MIN_SIGMA}"
    )
if FINAL_REFINE_FULL_SIGMA_FACTOR <= 0.0:
    raise ValueError(
        f"FINAL_REFINE_FULL_SIGMA_FACTOR must be > 0, got {FINAL_REFINE_FULL_SIGMA_FACTOR}"
    )
if FINAL_REFINE_FULL_MIN_SIGMA_AMP <= 0.0:
    raise ValueError(
        "FINAL_REFINE_FULL_MIN_SIGMA_AMP must be > 0, "
        f"got {FINAL_REFINE_FULL_MIN_SIGMA_AMP}"
    )
if FINAL_REFINE_FULL_SIGMA_FACTOR_AMP <= 0.0:
    raise ValueError(
        "FINAL_REFINE_FULL_SIGMA_FACTOR_AMP must be > 0, "
        f"got {FINAL_REFINE_FULL_SIGMA_FACTOR_AMP}"
    )
if (
    not np.isfinite(FINAL_REFINE_DIVERSITY_PHASE_RMS)
    or FINAL_REFINE_DIVERSITY_PHASE_RMS < 0.0
):
    raise ValueError(
        "FINAL_REFINE_DIVERSITY_PHASE_RMS must be finite and >= 0, "
        f"got {FINAL_REFINE_DIVERSITY_PHASE_RMS}"
    )
if (
    not np.isfinite(FINAL_REFINE_DIVERSITY_AMP_RMS)
    or FINAL_REFINE_DIVERSITY_AMP_RMS < 0.0
):
    raise ValueError(
        "FINAL_REFINE_DIVERSITY_AMP_RMS must be finite and >= 0, "
        f"got {FINAL_REFINE_DIVERSITY_AMP_RMS}"
    )
_valid_refine_modes = {"best", "elite_mean"}
if FINAL_REFINE_SEG_UPDATE_MODE not in _valid_refine_modes:
    raise ValueError(
        f"Unsupported FINAL_REFINE_SEG_UPDATE_MODE={FINAL_REFINE_SEG_UPDATE_MODE}, "
        f"expected one of {sorted(_valid_refine_modes)}"
    )
if FINAL_REFINE_FULL_UPDATE_MODE not in _valid_refine_modes:
    raise ValueError(
        f"Unsupported FINAL_REFINE_FULL_UPDATE_MODE={FINAL_REFINE_FULL_UPDATE_MODE}, "
        f"expected one of {sorted(_valid_refine_modes)}"
    )
if (
    not np.isfinite(FINAL_REFINE_SEG_ELITE_FRACTION)
    or FINAL_REFINE_SEG_ELITE_FRACTION <= 0.0
    or FINAL_REFINE_SEG_ELITE_FRACTION > 1.0
):
    raise ValueError(
        "FINAL_REFINE_SEG_ELITE_FRACTION must be in (0, 1], "
        f"got {FINAL_REFINE_SEG_ELITE_FRACTION}"
    )
if (
    not np.isfinite(FINAL_REFINE_FULL_ELITE_FRACTION)
    or FINAL_REFINE_FULL_ELITE_FRACTION <= 0.0
    or FINAL_REFINE_FULL_ELITE_FRACTION > 1.0
):
    raise ValueError(
        "FINAL_REFINE_FULL_ELITE_FRACTION must be in (0, 1], "
        f"got {FINAL_REFINE_FULL_ELITE_FRACTION}"
    )
if not np.isfinite(FINAL_REFINE_ELITE_WEIGHT_POWER) or FINAL_REFINE_ELITE_WEIGHT_POWER <= 0.0:
    raise ValueError(
        "FINAL_REFINE_ELITE_WEIGHT_POWER must be > 0 and finite, "
        f"got {FINAL_REFINE_ELITE_WEIGHT_POWER}"
    )
if FINAL_POLISH_ITERS < 0:
    raise ValueError(f"FINAL_POLISH_ITERS must be >= 0, got {FINAL_POLISH_ITERS}")
if FINAL_POLISH_PHASE_A0 <= 0.0 or FINAL_POLISH_PHASE_C0 <= 0.0:
    raise ValueError(
        "FINAL_POLISH_PHASE_A0 and FINAL_POLISH_PHASE_C0 must both be > 0."
    )
if FINAL_POLISH_AMP_A0 <= 0.0 or FINAL_POLISH_AMP_C0 <= 0.0:
    raise ValueError("FINAL_POLISH_AMP_A0 and FINAL_POLISH_AMP_C0 must both be > 0.")
if FINAL_POLISH_ALPHA <= 0.0 or FINAL_POLISH_GAMMA <= 0.0:
    raise ValueError("FINAL_POLISH_ALPHA and FINAL_POLISH_GAMMA must both be > 0.")
if FINAL_POLISH_PATIENCE < 0:
    raise ValueError(f"FINAL_POLISH_PATIENCE must be >= 0, got {FINAL_POLISH_PATIENCE}")
if FINAL_POLISH_LOG_INTERVAL <= 0:
    raise ValueError(
        f"FINAL_POLISH_LOG_INTERVAL must be > 0, got {FINAL_POLISH_LOG_INTERVAL}"
    )
if FINAL_POLISH_TOP_CANDIDATES < 0:
    raise ValueError(
        "FINAL_POLISH_TOP_CANDIDATES must be >= 0, "
        f"got {FINAL_POLISH_TOP_CANDIDATES}"
    )
if FINAL_POLISH_ACCEPT_MODE not in {"greedy", "spsa"}:
    raise ValueError(
        "FINAL_POLISH_ACCEPT_MODE must be one of {'greedy','spsa'}, "
        f"got {FINAL_POLISH_ACCEPT_MODE}"
    )
if FINAL_POLISH_SCOUT_ITERS < 0:
    raise ValueError(
        f"FINAL_POLISH_SCOUT_ITERS must be >= 0, got {FINAL_POLISH_SCOUT_ITERS}"
    )
if FINAL_POLISH_SCOUT_PATIENCE < 0:
    raise ValueError(
        "FINAL_POLISH_SCOUT_PATIENCE must be >= 0, "
        f"got {FINAL_POLISH_SCOUT_PATIENCE}"
    )
if FINAL_POLISH_FINAL_ITERS < 0:
    raise ValueError(
        f"FINAL_POLISH_FINAL_ITERS must be >= 0, got {FINAL_POLISH_FINAL_ITERS}"
    )
if FINAL_POLISH_FINAL_TOP_CENTERS <= 0:
    raise ValueError(
        "FINAL_POLISH_FINAL_TOP_CENTERS must be > 0, "
        f"got {FINAL_POLISH_FINAL_TOP_CENTERS}"
    )
if FINAL_POLISH_FINAL_RESTARTS <= 0:
    raise ValueError(
        "FINAL_POLISH_FINAL_RESTARTS must be > 0, "
        f"got {FINAL_POLISH_FINAL_RESTARTS}"
    )
if (
    not np.isfinite(FINAL_POLISH_RESTART_PHASE_JITTER)
    or FINAL_POLISH_RESTART_PHASE_JITTER < 0.0
):
    raise ValueError(
        "FINAL_POLISH_RESTART_PHASE_JITTER must be finite and >= 0, "
        f"got {FINAL_POLISH_RESTART_PHASE_JITTER}"
    )
if (
    not np.isfinite(FINAL_POLISH_RESTART_AMP_JITTER)
    or FINAL_POLISH_RESTART_AMP_JITTER < 0.0
):
    raise ValueError(
        "FINAL_POLISH_RESTART_AMP_JITTER must be finite and >= 0, "
        f"got {FINAL_POLISH_RESTART_AMP_JITTER}"
    )
if (
    not np.isfinite(FINAL_POLISH_RESTART_DECAY)
    or FINAL_POLISH_RESTART_DECAY <= 0.0
    or FINAL_POLISH_RESTART_DECAY > 1.0
):
    raise ValueError(
        "FINAL_POLISH_RESTART_DECAY must be in (0, 1], "
        f"got {FINAL_POLISH_RESTART_DECAY}"
    )
if FINAL_POLISH_EXTEND_BLOCK <= 0:
    raise ValueError(
        f"FINAL_POLISH_EXTEND_BLOCK must be > 0, got {FINAL_POLISH_EXTEND_BLOCK}"
    )
if FINAL_POLISH_MAX_ITERS <= 0:
    raise ValueError(
        f"FINAL_POLISH_MAX_ITERS must be > 0, got {FINAL_POLISH_MAX_ITERS}"
    )
if not np.isfinite(FINAL_POLISH_EXTEND_MIN_GAIN) or FINAL_POLISH_EXTEND_MIN_GAIN < 0.0:
    raise ValueError(
        "FINAL_POLISH_EXTEND_MIN_GAIN must be finite and >= 0, "
        f"got {FINAL_POLISH_EXTEND_MIN_GAIN}"
    )
if FINAL_COORD_POLISH_SWEEPS < 0:
    raise ValueError(
        f"FINAL_COORD_POLISH_SWEEPS must be >= 0, got {FINAL_COORD_POLISH_SWEEPS}"
    )
if FINAL_COORD_POLISH_MAX_MOVES <= 0:
    raise ValueError(
        "FINAL_COORD_POLISH_MAX_MOVES must be > 0, "
        f"got {FINAL_COORD_POLISH_MAX_MOVES}"
    )
if FINAL_COORD_POLISH_BATCH_SIZE <= 0:
    raise ValueError(
        "FINAL_COORD_POLISH_BATCH_SIZE must be > 0, "
        f"got {FINAL_COORD_POLISH_BATCH_SIZE}"
    )
if FINAL_COORD_POLISH_PHASE_STEP <= 0.0:
    raise ValueError(
        "FINAL_COORD_POLISH_PHASE_STEP must be > 0, "
        f"got {FINAL_COORD_POLISH_PHASE_STEP}"
    )
if FINAL_COORD_POLISH_PHASE_DECAY <= 0.0 or FINAL_COORD_POLISH_PHASE_DECAY > 1.0:
    raise ValueError(
        "FINAL_COORD_POLISH_PHASE_DECAY must be in (0, 1], "
        f"got {FINAL_COORD_POLISH_PHASE_DECAY}"
    )
if FINAL_COORD_POLISH_PHASE_MIN_STEP <= 0.0:
    raise ValueError(
        "FINAL_COORD_POLISH_PHASE_MIN_STEP must be > 0, "
        f"got {FINAL_COORD_POLISH_PHASE_MIN_STEP}"
    )
if FINAL_COORD_POLISH_AMP_STEP <= 0.0:
    raise ValueError(
        "FINAL_COORD_POLISH_AMP_STEP must be > 0, "
        f"got {FINAL_COORD_POLISH_AMP_STEP}"
    )
if FINAL_COORD_POLISH_AMP_DECAY <= 0.0 or FINAL_COORD_POLISH_AMP_DECAY > 1.0:
    raise ValueError(
        "FINAL_COORD_POLISH_AMP_DECAY must be in (0, 1], "
        f"got {FINAL_COORD_POLISH_AMP_DECAY}"
    )
if FINAL_COORD_POLISH_AMP_MIN_STEP <= 0.0:
    raise ValueError(
        "FINAL_COORD_POLISH_AMP_MIN_STEP must be > 0, "
        f"got {FINAL_COORD_POLISH_AMP_MIN_STEP}"
    )
if not np.isfinite(FINAL_COORD_POLISH_MIN_GAIN) or FINAL_COORD_POLISH_MIN_GAIN < 0.0:
    raise ValueError(
        "FINAL_COORD_POLISH_MIN_GAIN must be finite and >= 0, "
        f"got {FINAL_COORD_POLISH_MIN_GAIN}"
    )
if FINAL_COORD_POLISH_TOP_CANDIDATES <= 0:
    raise ValueError(
        "FINAL_COORD_POLISH_TOP_CANDIDATES must be > 0, "
        f"got {FINAL_COORD_POLISH_TOP_CANDIDATES}"
    )
if FINAL_COORD_POLISH_EMPTY_SWEEP_PATIENCE < 0:
    raise ValueError(
        "FINAL_COORD_POLISH_EMPTY_SWEEP_PATIENCE must be >= 0, "
        f"got {FINAL_COORD_POLISH_EMPTY_SWEEP_PATIENCE}"
    )
if (
    not np.isfinite(FINAL_COORD_POLISH_RESERVE_SEC)
    or FINAL_COORD_POLISH_RESERVE_SEC < 0.0
):
    raise ValueError(
        "FINAL_COORD_POLISH_RESERVE_SEC must be finite and >= 0, "
        f"got {FINAL_COORD_POLISH_RESERVE_SEC}"
    )
if not np.isfinite(POST_RL_BUDGET_SEC) or POST_RL_BUDGET_SEC < 0.0:
    raise ValueError(
        f"POST_RL_BUDGET_SEC must be finite and >= 0, got {POST_RL_BUDGET_SEC}"
    )
if not np.isfinite(POST_RL_STOP_BUFFER_SEC) or POST_RL_STOP_BUFFER_SEC < 0.0:
    raise ValueError(
        "POST_RL_STOP_BUFFER_SEC must be finite and >= 0, "
        f"got {POST_RL_STOP_BUFFER_SEC}"
    )
if not np.isfinite(POST_RL_MIN_STAGE_SEC) or POST_RL_MIN_STAGE_SEC < 0.0:
    raise ValueError(
        "POST_RL_MIN_STAGE_SEC must be finite and >= 0, "
        f"got {POST_RL_MIN_STAGE_SEC}"
    )
if (
    FINAL_COORD_POLISH_ENABLE
    and POST_RL_BUDGET_SEC > 0.0
    and FINAL_COORD_POLISH_RESERVE_SEC <= 0.0
):
    FINAL_COORD_POLISH_RESERVE_SEC = min(
        900.0, max(120.0, 0.15 * POST_RL_BUDGET_SEC)
    )
POST_RL_COORD_RESERVE_SEC = (
    FINAL_COORD_POLISH_RESERVE_SEC if FINAL_COORD_POLISH_ENABLE else 0.0
)
if TRAIN_FID_SCREEN_START_EPOCH < 0:
    raise ValueError(
        f"TRAIN_FID_SCREEN_START_EPOCH must be >= 0, got {TRAIN_FID_SCREEN_START_EPOCH}"
    )
if TRAIN_FID_SCREEN_INTERVAL <= 0:
    raise ValueError(
        f"TRAIN_FID_SCREEN_INTERVAL must be > 0, got {TRAIN_FID_SCREEN_INTERVAL}"
    )
if TRAIN_FID_SCREEN_TOPK <= 0:
    raise ValueError(f"TRAIN_FID_SCREEN_TOPK must be > 0, got {TRAIN_FID_SCREEN_TOPK}")
if TRAIN_FID_SCREEN_RECORDS < 0:
    raise ValueError(
        f"TRAIN_FID_SCREEN_RECORDS must be >= 0, got {TRAIN_FID_SCREEN_RECORDS}"
    )
if (
    not np.isfinite(INIT_PULSE_BLEND)
    or INIT_PULSE_BLEND < 0.0
    or INIT_PULSE_BLEND > 1.0
):
    raise ValueError(f"INIT_PULSE_BLEND must be in [0,1], got {INIT_PULSE_BLEND}")
if (
    not np.isfinite(FINAL_REFINE_CHECKPOINT_BLEND)
    or FINAL_REFINE_CHECKPOINT_BLEND < 0.0
    or FINAL_REFINE_CHECKPOINT_BLEND > 1.0
):
    raise ValueError(
        "FINAL_REFINE_CHECKPOINT_BLEND must be in [0,1], "
        f"got {FINAL_REFINE_CHECKPOINT_BLEND}"
    )

_gkp_diag = gkp_target_fock_statistics(
    delta=GKP_DELTA,
    n_boson=N_BOSON,
    logical=GKP_LOGICAL,
    rel_phase=GKP_REL_PHASE,
    squeeze_r=GKP_SQUEEZE_R,
    envelope_kappa=GKP_ENVELOPE_KAPPA,
    lattice_trunc=GKP_LATTICE_TRUNC,
)
GKP_TARGET_MEAN_N = float(_gkp_diag["mean_n"])
GKP_TARGET_TAIL_START = int(_gkp_diag["tail_start"])
GKP_TARGET_TAIL_MASS = float(_gkp_diag["tail_mass"])
GKP_TARGET_EDGE_PROB = float(_gkp_diag["edge_prob"])
if not FAST_SMOKE and GKP_TARGET_TAIL_MASS > GKP_TAIL_ERROR and not ALLOW_LOW_N_BOSON:
    raise ValueError(
        "N_BOSON is too low for the current GKP target: "
        f"tail mass (n>={GKP_TARGET_TAIL_START})={GKP_TARGET_TAIL_MASS:.3e} > {GKP_TAIL_ERROR:.3e}. "
        "Increase N_BOSON or set ALLOW_LOW_N_BOSON=1 to override."
    )


def _build_characteristic_distribution(grid_size, gkp_delta):
    def _importance_weights(target_values):
        # For symmetric targets (e.g., GKP |0>), paper-aligned importance
        # sampling uses P(alpha) proportional to |chi_target(alpha)|^2.
        mags = np.abs(np.asarray(target_values, dtype=complex))
        weights = mags ** CHAR_IMPORTANCE_POWER
        if CHAR_UNIFORM_MIX > 0.0:
            weights = (1.0 - CHAR_UNIFORM_MIX) * weights + CHAR_UNIFORM_MIX * np.ones_like(
                weights
            )
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            return np.full(len(weights), 1.0 / max(1, len(weights)), dtype=float)
        return np.asarray(weights / total, dtype=float)

    all_points = []
    all_targets = []
    all_weights = []
    all_areas = []
    n_scales = len(CHAR_ALPHA_SCALES)
    for alpha_scale in CHAR_ALPHA_SCALES:
        points_i, target_i, weights_i, area_i = prepare_characteristic_distribution(
            alpha_cat=gkp_delta,
            gkp_delta=gkp_delta,
            n_boson=N_BOSON,
            extent=SAMPLE_EXTENT,
            grid_size=grid_size,
            cat_parity=GKP_LOGICAL,
            gkp_logical=GKP_LOGICAL,
            mix_uniform=0.0,
            alpha_scale=alpha_scale,
            cat_phase=GKP_REL_PHASE,
            gkp_squeeze_r=GKP_SQUEEZE_R,
            gkp_envelope_kappa=GKP_ENVELOPE_KAPPA,
            gkp_lattice_trunc=GKP_LATTICE_TRUNC,
        )
        all_points.extend(points_i)
        all_targets.append(target_i)
        all_weights.append(_importance_weights(target_i) / float(n_scales))
        all_areas.append(area_i)
    return (
        all_points,
        np.concatenate(all_targets),
        np.concatenate(all_weights),
        float(np.mean(all_areas)),
    )


def _overlap_norm_from_target(target_values):
    # For importance-sampled overlap estimators, E[|chi|^2 / p] reduces to
    # sum(|chi|^2) over the discrete bank, independent of p.
    target_values = np.asarray(target_values, dtype=complex)
    norm = float(np.sum(np.abs(target_values) ** 2))
    if not np.isfinite(norm) or norm <= 0.0:
        return 1.0
    return norm


def _build_stage1_topk_indices(score, count):
    # For stage-1 warmup, keep strongest informative points globally.
    if count <= 0:
        return np.array([], dtype=int)
    return np.argsort(score)[-count:]


def _gkp_lattice_points(order):
    a = np.sqrt(np.pi / 2.0)
    pts = []
    for m in range(-order, order + 1):
        for n in range(-order, order + 1):
            if m == 0 and n == 0:
                continue
            pts.append((m * a) + 1j * (n * a))
    return np.asarray(pts, dtype=complex)


def _nearest_lattice_indices(char_points, lattice_points):
    if lattice_points.size == 0 or len(char_points) == 0:
        return np.array([], dtype=int)
    arr = np.asarray(char_points, dtype=complex)
    idx = []
    for lp in lattice_points:
        i = int(np.argmin(np.abs(arr - lp)))
        idx.append(i)
    return np.unique(np.asarray(idx, dtype=int))


def _prepare_sampling_bank(gkp_delta):
    char_points, char_target, char_weights, char_area = _build_characteristic_distribution(
        CHAR_GRID_SIZE,
        gkp_delta=gkp_delta,
    )
    char_norm = characteristic_norm(char_target, char_area)
    char_overlap_norm = _overlap_norm_from_target(char_target)
    char_radii = np.abs(np.asarray(char_points))
    topk_count = min(TRAIN_POINTS_STAGE1, len(char_points))
    target_mag = np.abs(char_target) ** CHAR_IMPORTANCE_POWER
    if CHAR_START_MODE == "topk":
        score = target_mag
    else:
        radii = np.maximum(char_radii, 1e-6)
        score = target_mag * (radii ** CHAR_RADIAL_EXP)
    lattice_points = _gkp_lattice_points(GKP_LATTICE_ORDER)
    lattice_idx = _nearest_lattice_indices(char_points, lattice_points)
    if lattice_idx.size > 0 and GKP_LATTICE_TOPK_BOOST > 0.0:
        score = np.asarray(score, dtype=float).copy()
        score[lattice_idx] *= (1.0 + GKP_LATTICE_TOPK_BOOST)
    topk_idx = _build_stage1_topk_indices(score, topk_count)
    topk_points = [char_points[i] for i in topk_idx]
    topk_target = char_target[topk_idx]
    if topk_count > 0:
        topk_weights = np.full(topk_count, 1.0 / topk_count, dtype=float)
        topk_norm = characteristic_norm(topk_target, char_area)
        topk_overlap_norm = _overlap_norm_from_target(topk_target)
    else:
        topk_weights = np.empty((0,), dtype=float)
        topk_norm = 1.0
        topk_overlap_norm = 1.0
    lattice_probs = np.zeros(len(char_points), dtype=float)
    if lattice_idx.size > 0:
        local = target_mag[lattice_idx].astype(float)
        local_sum = float(np.sum(local))
        if local_sum <= 0.0 or not np.isfinite(local_sum):
            local = np.ones_like(local)
            local_sum = float(np.sum(local))
        lattice_probs[lattice_idx] = local / local_sum
    else:
        lattice_probs = np.asarray(char_weights, dtype=float).copy()
    lattice_probs = np.maximum(lattice_probs, 1e-12)
    lattice_probs = lattice_probs / float(np.sum(lattice_probs))
    return {
        "gkp_delta": float(gkp_delta),
        "char_points": char_points,
        "char_target": char_target,
        "char_weights": char_weights,
        "char_area": char_area,
        "char_norm": char_norm,
        "char_overlap_norm": char_overlap_norm,
        "char_radii": char_radii,
        "topk_points": topk_points,
        "topk_target": topk_target,
        "topk_weights": topk_weights,
        "topk_norm": topk_norm,
        "topk_overlap_norm": topk_overlap_norm,
        "lattice_idx": lattice_idx,
        "lattice_points": lattice_points,
        "lattice_probs": lattice_probs,
    }


def _train_delta_for_epoch(epoch):
    if len(GKP_DELTA_CURRICULUM) == 1:
        return float(GKP_DELTA_CURRICULUM[0])
    stage = 0
    for boundary in GKP_DELTA_CURRICULUM_EPOCHS:
        if epoch < boundary:
            break
        stage += 1
    stage = min(stage, len(GKP_DELTA_CURRICULUM) - 1)
    return float(GKP_DELTA_CURRICULUM[stage])


_train_delta_values = []
for _d in GKP_DELTA_CURRICULUM:
    if _d not in _train_delta_values:
        _train_delta_values.append(_d)
if GKP_DELTA not in _train_delta_values:
    _train_delta_values.append(GKP_DELTA)

TRAIN_BANKS = {float(d): _prepare_sampling_bank(float(d)) for d in _train_delta_values}
FINAL_BANK = TRAIN_BANKS[float(GKP_DELTA)]

FINAL_POINTS, FINAL_TARGET, FINAL_WEIGHTS, FINAL_AREA = prepare_characteristic_distribution(
    alpha_cat=GKP_DELTA,
    gkp_delta=GKP_DELTA,
    n_boson=N_BOSON,
    extent=SAMPLE_EXTENT,
    grid_size=FINAL_GRID_SIZE,
    cat_parity=GKP_LOGICAL,
    gkp_logical=GKP_LOGICAL,
    mix_uniform=0.0,
    alpha_scale=CHAR_ALPHA_SCALE,
    cat_phase=GKP_REL_PHASE,
    gkp_squeeze_r=GKP_SQUEEZE_R,
    gkp_envelope_kappa=GKP_ENVELOPE_KAPPA,
    gkp_lattice_trunc=GKP_LATTICE_TRUNC,
)
_final_mags = np.abs(np.asarray(FINAL_TARGET, dtype=complex))
FINAL_WEIGHTS = _final_mags ** CHAR_IMPORTANCE_POWER
if CHAR_UNIFORM_MIX > 0.0:
    FINAL_WEIGHTS = (1.0 - CHAR_UNIFORM_MIX) * FINAL_WEIGHTS + CHAR_UNIFORM_MIX * np.ones_like(
        FINAL_WEIGHTS
    )
_final_w_sum = float(np.sum(FINAL_WEIGHTS))
if not np.isfinite(_final_w_sum) or _final_w_sum <= 0.0:
    FINAL_WEIGHTS = np.full_like(FINAL_WEIGHTS, 1.0 / max(1, len(FINAL_WEIGHTS)), dtype=float)
else:
    FINAL_WEIGHTS = np.asarray(FINAL_WEIGHTS / _final_w_sum, dtype=float)
FINAL_NORM = characteristic_norm(FINAL_TARGET, FINAL_AREA)
FINAL_OVERLAP_NORM = _overlap_norm_from_target(FINAL_TARGET)

logger.info(
    "Characteristic sampling: start_mode=%s alpha_scales=%s radial_exp=%.2f gkp_phase=%s",
    CHAR_START_MODE,
    ",".join(f"{v:.3f}" for v in CHAR_ALPHA_SCALES),
    CHAR_RADIAL_EXP,
    "none" if GKP_REL_PHASE is None else f"{GKP_REL_PHASE:.3f}",
)
logger.info(
    "GKP params: delta=%.4f logical=%s squeeze_r=%s envelope_kappa=%s lattice_trunc=%d",
    GKP_DELTA,
    GKP_LOGICAL,
    "auto(-log(delta))" if GKP_SQUEEZE_R is None else f"{GKP_SQUEEZE_R:.4f}",
    "auto(delta)" if GKP_ENVELOPE_KAPPA is None else f"{GKP_ENVELOPE_KAPPA:.4f}",
    GKP_LATTICE_TRUNC,
)
logger.info(
    "GKP truncation check: N_BOSON=%d mean_n=%.4f tail(n>=%d)=%.3e edge_prob=%.3e",
    N_BOSON,
    GKP_TARGET_MEAN_N,
    GKP_TARGET_TAIL_START,
    GKP_TARGET_TAIL_MASS,
    GKP_TARGET_EDGE_PROB,
)
if GKP_TARGET_TAIL_MASS > GKP_TAIL_WARN:
    logger.warning(
        "Target has non-negligible high-Fock tail mass (%.3e > %.3e). "
        "Consider increasing N_BOSON for higher-fidelity optimization.",
        GKP_TARGET_TAIL_MASS,
        GKP_TAIL_WARN,
    )
if len(GKP_DELTA_CURRICULUM) > 1:
    logger.info(
        "GKP delta curriculum: deltas=%s switch_epochs=%s final_eval_delta=%.4f",
        ",".join(f"{v:.4f}" for v in GKP_DELTA_CURRICULUM),
        ",".join(str(v) for v in GKP_DELTA_CURRICULUM_EPOCHS),
        GKP_DELTA,
    )
logger.info(
    "Characteristic reward objective schedule: base=%s stage2=%s switch_train_epoch=%d (raw=%d) | stage epochs: %d -> %d -> end",
    CHAR_REWARD_OBJECTIVE,
    CHAR_REWARD_OBJECTIVE_STAGE2,
    CHAR_REWARD_SWITCH_EPOCH_EFFECTIVE,
    CHAR_REWARD_SWITCH_EPOCH,
    TRAIN_STAGE1_EPOCHS,
    TRAIN_STAGE2_EPOCHS,
)
if CHAR_REWARD_SWITCH_AUTOCONFIG_NOTE:
    logger.warning(CHAR_REWARD_SWITCH_AUTOCONFIG_NOTE)
logger.info(
    "Reward switch guard: min_best_eval=%.3f force_after_eval=%d patience_eval=%d min_gain=%.4f allow_revert=%s",
    CHAR_REWARD_SWITCH_MIN_BEST_EVAL,
    CHAR_REWARD_SWITCH_FORCE_AFTER_EVAL_PATIENCE,
    CHAR_REWARD_STAGE2_PATIENCE_EVAL,
    CHAR_REWARD_STAGE2_MIN_GAIN,
    CHAR_REWARD_STAGE2_ALLOW_REVERT,
)
logger.info(
    "Characteristic point sampler: mode=%s radial_bins=%d uniform_mix=%.2f importance_power=%.2f",
    CHAR_SAMPLER_MODE,
    CHAR_RADIAL_BINS,
    CHAR_UNIFORM_MIX,
    CHAR_IMPORTANCE_POWER,
)
logger.info(
    "GKP lattice-aware sampling: mix_start=%.2f mix_end=%.2f anneal_epochs=%d topk_boost=%.2f lattice_order=%d",
    GKP_LATTICE_MIX_START,
    GKP_LATTICE_MIX_END,
    GKP_LATTICE_MIX_ANNEAL_EPOCHS,
    GKP_LATTICE_TOPK_BOOST,
    GKP_LATTICE_ORDER,
)
logger.info(
    "GKP lattice anchors in final bank: %d",
    int(FINAL_BANK["lattice_idx"].size),
)
logger.info(
    "Action clipping: phase_clip=%.3f amp_range=[%.3f, %.3f]",
    PHASE_CLIP,
    AMP_MIN,
    AMP_MAX,
)
logger.info(
    "Phase projection: periodic_wrap=%s (requested=%s, clip_is_pi=%s)",
    _PHASE_WRAP_ENABLED,
    USE_PERIODIC_PHASE_PROJECTION,
    np.isclose(PHASE_CLIP, np.pi, atol=1.0e-9),
)
logger.info(
    "Pulse timing: total_duration=%.3f t_step=%.6f n_steps=%d",
    TOTAL_DURATION,
    T_STEP,
    N_STEPS,
)
logger.info(
    "Characteristic plotting extent: %.3f (sampling extent: %.3f)",
    PLOT_EXTENT,
    SAMPLE_EXTENT,
)
logger.info(
    "Final refinement setup: samples=%d rounds=%d topk=%d top_eval_centers=%d scale=%.3f decay=%.3f min_sigma=%.3f amp_opt=%s min_sigma_amp=%.3f init_sigma_amp=%.3f amp_start_round=%d use_loc_center=%s use_train_center=%s",
    FINAL_REFINE_SAMPLES,
    FINAL_REFINE_ROUNDS,
    FINAL_REFINE_TOPK,
    FINAL_REFINE_TOP_EVAL_CENTERS,
    FINAL_REFINE_SCALE,
    FINAL_REFINE_DECAY,
    FINAL_REFINE_MIN_SIGMA,
    FINAL_REFINE_ENABLE_AMP,
    FINAL_REFINE_MIN_SIGMA_AMP,
    FINAL_REFINE_INIT_SIGMA_AMP,
    FINAL_REFINE_AMP_START_ROUND,
    FINAL_REFINE_USE_LOC_CENTER,
    FINAL_REFINE_USE_TRAIN_CENTER,
)
logger.info(
    "Extra refinement centers: warmstart=%s init_npz=%s init_blend=%.3f checkpoint=%s checkpoint_npz=%s checkpoint_blend=%.3f",
    FINAL_REFINE_USE_WARMSTART_CENTER,
    INIT_PULSES_NPZ if INIT_PULSES_NPZ else "none",
    INIT_PULSE_BLEND,
    FINAL_REFINE_USE_CHECKPOINT_CENTER,
    FINAL_REFINE_CHECKPOINT_PULSES if FINAL_REFINE_CHECKPOINT_PULSES else "none",
    FINAL_REFINE_CHECKPOINT_BLEND,
)
logger.info(
    "Full-step refinement: enabled=%s samples=%d rounds=%d topk=%d centers=%d scale=%.3f decay=%.3f phase_sigma_factor=%.3f phase_min_sigma=%.4f amp_opt=%s amp_sigma_factor=%.3f amp_min_sigma=%.4f",
    FINAL_REFINE_FULL_STEPS,
    FINAL_REFINE_FULL_SAMPLES,
    FINAL_REFINE_FULL_ROUNDS,
    FINAL_REFINE_FULL_TOPK,
    FINAL_REFINE_FULL_CANDIDATES,
    FINAL_REFINE_FULL_SCALE,
    FINAL_REFINE_FULL_DECAY,
    FINAL_REFINE_FULL_SIGMA_FACTOR,
    FINAL_REFINE_FULL_MIN_SIGMA,
    FINAL_REFINE_FULL_ENABLE_AMP,
    FINAL_REFINE_FULL_SIGMA_FACTOR_AMP,
    FINAL_REFINE_FULL_MIN_SIGMA_AMP,
)
logger.info(
    "Refinement center update: seg_mode=%s seg_elite_frac=%.2f full_mode=%s full_elite_frac=%.2f elite_weight_power=%.2f",
    FINAL_REFINE_SEG_UPDATE_MODE,
    FINAL_REFINE_SEG_ELITE_FRACTION,
    FINAL_REFINE_FULL_UPDATE_MODE,
    FINAL_REFINE_FULL_ELITE_FRACTION,
    FINAL_REFINE_ELITE_WEIGHT_POWER,
)
logger.info(
    "Full-step center diversity: enabled=%s phase_rms>=%.4f or amp_rms>=%.4f",
    FINAL_REFINE_DIVERSE_CENTERS,
    FINAL_REFINE_DIVERSITY_PHASE_RMS,
    FINAL_REFINE_DIVERSITY_AMP_RMS,
)
logger.info(
    "Final SPSA polish: enabled=%s iters=%d include_amp=%s top_candidates=%d mode=%s phase(a0=%.4f,c0=%.4f) amp(a0=%.4f,c0=%.4f) alpha=%.3f gamma=%.3f patience=%d",
    FINAL_POLISH_ENABLE,
    FINAL_POLISH_ITERS,
    FINAL_POLISH_INCLUDE_AMP,
    FINAL_POLISH_TOP_CANDIDATES,
    FINAL_POLISH_ACCEPT_MODE,
    FINAL_POLISH_PHASE_A0,
    FINAL_POLISH_PHASE_C0,
    FINAL_POLISH_AMP_A0,
    FINAL_POLISH_AMP_C0,
    FINAL_POLISH_ALPHA,
    FINAL_POLISH_GAMMA,
    FINAL_POLISH_PATIENCE,
)
logger.info(
    "Final SPSA schedule: scout_iters=%d scout_patience=%d final_iters=%d",
    FINAL_POLISH_SCOUT_ITERS,
    FINAL_POLISH_SCOUT_PATIENCE,
    FINAL_POLISH_FINAL_ITERS,
)
logger.info(
    "Final SPSA continuation portfolio: top_centers=%d restarts=%d phase_jitter=%.6g amp_jitter=%.6g decay=%.3f",
    FINAL_POLISH_FINAL_TOP_CENTERS,
    FINAL_POLISH_FINAL_RESTARTS,
    FINAL_POLISH_RESTART_PHASE_JITTER,
    FINAL_POLISH_RESTART_AMP_JITTER,
    FINAL_POLISH_RESTART_DECAY,
)
logger.info(
    "Final SPSA auto-extend: enabled=%s block=%d max_iters=%d min_gain=%.6g",
    FINAL_POLISH_AUTO_EXTEND,
    FINAL_POLISH_EXTEND_BLOCK,
    FINAL_POLISH_MAX_ITERS,
    FINAL_POLISH_EXTEND_MIN_GAIN,
)
logger.info(
    "Coordinate polish: enabled=%s include_amp=%s sweeps=%d max_moves=%d batch_size=%d top_candidates=%d "
    "phase(step=%.6g,decay=%.3f,min=%.6g) amp(step=%.6g,decay=%.3f,min=%.6g) min_gain=%.6g",
    FINAL_COORD_POLISH_ENABLE,
    FINAL_COORD_POLISH_INCLUDE_AMP,
    FINAL_COORD_POLISH_SWEEPS,
    FINAL_COORD_POLISH_MAX_MOVES,
    FINAL_COORD_POLISH_BATCH_SIZE,
    FINAL_COORD_POLISH_TOP_CANDIDATES,
    FINAL_COORD_POLISH_PHASE_STEP,
    FINAL_COORD_POLISH_PHASE_DECAY,
    FINAL_COORD_POLISH_PHASE_MIN_STEP,
    FINAL_COORD_POLISH_AMP_STEP,
    FINAL_COORD_POLISH_AMP_DECAY,
    FINAL_COORD_POLISH_AMP_MIN_STEP,
    FINAL_COORD_POLISH_MIN_GAIN,
)
logger.info(
    "Coordinate polish empty-sweep patience: %d",
    FINAL_COORD_POLISH_EMPTY_SWEEP_PATIENCE,
)
logger.info(
    "Post-RL budget guard: budget_sec=%.1f stop_buffer_sec=%.1f min_stage_sec=%.1f coord_reserve_sec=%.1f",
    POST_RL_BUDGET_SEC,
    POST_RL_STOP_BUFFER_SEC,
    POST_RL_MIN_STAGE_SEC,
    POST_RL_COORD_RESERVE_SEC,
)
logger.info(
    "Characteristic reward normalization: fixed_overlap_norm=%s final_overlap_norm=%.6f",
    CHAR_USE_FIXED_REWARD_NORM,
    FINAL_OVERLAP_NORM,
)
logger.info(
    "Train-batch fidelity screening: enabled=%s start_epoch=%d interval=%d topk=%d records=%d use_train_fid_center=%s",
    TRAIN_FID_SCREEN_ENABLE,
    TRAIN_FID_SCREEN_START_EPOCH,
    TRAIN_FID_SCREEN_INTERVAL,
    TRAIN_FID_SCREEN_TOPK,
    TRAIN_FID_SCREEN_RECORDS,
    FINAL_REFINE_USE_TRAIN_FID_CENTER,
)


def _smoothness_penalty(phi_r, phi_b, amp_r, amp_b):
    axis = 1 if np.ndim(phi_r) > 1 else 0
    dphi_r = np.diff(phi_r, axis=axis)
    dphi_b = np.diff(phi_b, axis=axis)
    damp_r = np.diff(amp_r, axis=axis)
    damp_b = np.diff(amp_b, axis=axis)
    phi_pen = 0.5 * (np.mean(dphi_r ** 2, axis=axis) + np.mean(dphi_b ** 2, axis=axis))
    amp_pen = 0.5 * (np.mean(damp_r ** 2, axis=axis) + np.mean(damp_b ** 2, axis=axis))
    return SMOOTH_PHI_WEIGHT * phi_pen + SMOOTH_AMP_WEIGHT * amp_pen


def _log_action_stats(tag, phi_r, phi_b, amp_r, amp_b):
    def _stats(arr):
        arr = np.asarray(arr)
        return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))

    for name, arr in [
        ("phi_r", phi_r),
        ("phi_b", phi_b),
        ("amp_r", amp_r),
        ("amp_b", amp_b),
    ]:
        mean, std, vmin, vmax = _stats(arr)
        logger.info(
            "%s %s stats: mean=%.4f std=%.4f min=%.4f max=%.4f",
            tag,
            name,
            mean,
            std,
            vmin,
            vmax,
        )


def _log_batch_diversity(tag, phi_r, phi_b, amp_r, amp_b):
    def _batch_stats(arr):
        arr = np.asarray(arr)
        if arr.ndim < 2:
            return 0.0, 0.0
        std_over_batch = np.std(arr, axis=0)
        return float(np.mean(std_over_batch)), float(np.max(std_over_batch))

    for name, arr in [
        ("phi_r", phi_r),
        ("phi_b", phi_b),
        ("amp_r", amp_r),
        ("amp_b", amp_b),
    ]:
        mean_std, max_std = _batch_stats(arr)
        logger.info(
            "%s %s batch-std: mean=%.6f max=%.6f",
            tag,
            name,
            mean_std,
            max_std,
        )


def _project_phase(values):
    arr = np.asarray(values, dtype=float)
    if _PHASE_WRAP_ENABLED:
        return np.angle(np.exp(1j * arr))
    return np.clip(arr, -PHASE_CLIP, PHASE_CLIP)


def _project_phase_scalar(value):
    val = float(value)
    if _PHASE_WRAP_ENABLED:
        return float(np.angle(np.exp(1j * val)))
    return float(np.clip(val, -PHASE_CLIP, PHASE_CLIP))


def _segment_phase_from_full(arr):
    vals = np.asarray(arr, dtype=float).reshape(-1)
    if vals.size == N_SEGMENTS:
        return vals.copy()
    if N_SEGMENTS % vals.size == 0:
        rep = N_SEGMENTS // vals.size
        return np.repeat(vals, rep)
    if vals.size % N_SEGMENTS == 0:
        seg_len = vals.size // N_SEGMENTS
        reshaped = vals.reshape(N_SEGMENTS, seg_len)
        return np.angle(np.mean(np.exp(1j * reshaped), axis=1))

    src = np.arange(vals.size, dtype=float)
    dst = (np.arange(N_SEGMENTS, dtype=float) + 0.5) * (vals.size / N_SEGMENTS) - 0.5
    dst = np.clip(dst, 0.0, max(float(vals.size - 1), 0.0))
    unwrapped = np.unwrap(vals)
    interp = np.interp(dst, src, unwrapped)
    return np.angle(np.exp(1j * interp))


def _segment_amp_from_full(arr):
    vals = np.asarray(arr, dtype=float).reshape(-1)
    if vals.size == N_SEGMENTS:
        return vals.copy()
    if N_SEGMENTS % vals.size == 0:
        rep = N_SEGMENTS // vals.size
        return np.repeat(vals, rep)
    if vals.size % N_SEGMENTS == 0:
        seg_len = vals.size // N_SEGMENTS
        reshaped = vals.reshape(N_SEGMENTS, seg_len)
        return np.mean(reshaped, axis=1)

    src = np.arange(vals.size, dtype=float)
    dst = (np.arange(N_SEGMENTS, dtype=float) + 0.5) * (vals.size / N_SEGMENTS) - 0.5
    dst = np.clip(dst, 0.0, max(float(vals.size - 1), 0.0))
    return np.interp(dst, src, vals)


def _resolve_optional_npz_path(path_raw):
    path_raw = str(path_raw).strip()
    if path_raw == "":
        return ""
    if os.path.isabs(path_raw):
        return path_raw
    return os.path.join(os.getcwd(), path_raw)


def _load_pulse_center_from_npz(path_raw, blend, label):
    resolved = _resolve_optional_npz_path(path_raw)
    if resolved == "":
        return None
    if not os.path.exists(resolved):
        logger.warning("Skip %s center: pulse file not found: %s", label, resolved)
        return None
    try:
        with np.load(resolved) as data:
            req = ("phi_r", "phi_b", "amp_r", "amp_b")
            missing = [k for k in req if k not in data]
            if missing:
                logger.warning(
                    "Skip %s center: missing keys %s in %s",
                    label,
                    missing,
                    resolved,
                )
                return None
            phi_r_raw = np.asarray(data["phi_r"], dtype=float).reshape(-1)
            phi_b_raw = np.asarray(data["phi_b"], dtype=float).reshape(-1)
            amp_r_raw = np.asarray(data["amp_r"], dtype=float).reshape(-1)
            amp_b_raw = np.asarray(data["amp_b"], dtype=float).reshape(-1)
    except Exception as exc:
        logger.warning("Skip %s center: failed loading %s (%s)", label, resolved, exc)
        return None

    blend = float(blend)
    phi_r_seg = _project_phase(blend * _segment_phase_from_full(phi_r_raw))
    phi_b_seg = _project_phase(blend * _segment_phase_from_full(phi_b_raw))
    amp_r_seg = np.clip((1.0 - blend) + blend * _segment_amp_from_full(amp_r_raw), AMP_MIN, AMP_MAX)
    amp_b_seg = np.clip((1.0 - blend) + blend * _segment_amp_from_full(amp_b_raw), AMP_MIN, AMP_MAX)
    logger.info(
        "Loaded %s center from %s (blend=%.3f, lens phi_r=%d phi_b=%d amp_r=%d amp_b=%d -> segments=%d)",
        label,
        resolved,
        blend,
        phi_r_raw.size,
        phi_b_raw.size,
        amp_r_raw.size,
        amp_b_raw.size,
        N_SEGMENTS,
    )
    return phi_r_seg, phi_b_seg, amp_r_seg, amp_b_seg


def _effective_train_epoch(epoch, epoch_type):
    if epoch_type == "final":
        return NUM_EPOCHS_HINT
    if epoch_type == "evaluation":
        return int(epoch) * max(1, EVAL_INTERVAL_HINT)
    return int(epoch)


_reward_schedule_state = {
    "active": CHAR_REWARD_OBJECTIVE,
    "switched": False,
    "reverted": False,
    "switch_forced": False,
    "switch_eval_epoch": None,
    "anchor_best_eval": -np.inf,
    "stage2_best_eval": -np.inf,
    "defer_eval_count": 0,
}


def _update_reward_objective(epoch, epoch_type, best_eval_fidelity):
    active = _reward_schedule_state["active"]
    if CHAR_REWARD_OBJECTIVE_STAGE2 == CHAR_REWARD_OBJECTIVE:
        return active

    effective_train_epoch = _effective_train_epoch(epoch, epoch_type)
    reached_switch_window = (
        not _reward_schedule_state["switched"]
        and CHAR_REWARD_SWITCH_EPOCH_EFFECTIVE >= 0
        and epoch_type == "evaluation"
        and effective_train_epoch >= CHAR_REWARD_SWITCH_EPOCH_EFFECTIVE
    )
    can_switch = reached_switch_window and (
        best_eval_fidelity >= CHAR_REWARD_SWITCH_MIN_BEST_EVAL
    )
    force_switch = False
    if reached_switch_window and not can_switch:
        _reward_schedule_state["defer_eval_count"] += 1
        force_switch = (
            CHAR_REWARD_SWITCH_FORCE_AFTER_EVAL_PATIENCE >= 0
            and _reward_schedule_state["defer_eval_count"]
            >= CHAR_REWARD_SWITCH_FORCE_AFTER_EVAL_PATIENCE
        )
    if can_switch or force_switch:
        _reward_schedule_state["active"] = CHAR_REWARD_OBJECTIVE_STAGE2
        _reward_schedule_state["switched"] = True
        _reward_schedule_state["switch_forced"] = bool(force_switch and not can_switch)
        _reward_schedule_state["switch_eval_epoch"] = int(epoch)
        _reward_schedule_state["anchor_best_eval"] = float(best_eval_fidelity)
        _reward_schedule_state["stage2_best_eval"] = float(best_eval_fidelity)
        return _reward_schedule_state["active"]

    if (
        _reward_schedule_state["switched"]
        and (not _reward_schedule_state["reverted"])
        and epoch_type == "evaluation"
    ):
        _reward_schedule_state["stage2_best_eval"] = max(
            float(_reward_schedule_state["stage2_best_eval"]),
            float(best_eval_fidelity),
        )
        waited_eval_epochs = int(epoch) - int(_reward_schedule_state["switch_eval_epoch"])
        if (
            CHAR_REWARD_STAGE2_ALLOW_REVERT
            and waited_eval_epochs >= CHAR_REWARD_STAGE2_PATIENCE_EVAL
            and _reward_schedule_state["stage2_best_eval"]
            < (_reward_schedule_state["anchor_best_eval"] + CHAR_REWARD_STAGE2_MIN_GAIN)
        ):
            _reward_schedule_state["active"] = CHAR_REWARD_OBJECTIVE
            _reward_schedule_state["reverted"] = True

    return _reward_schedule_state["active"]


def _lattice_mix_for_epoch(epoch):
    if GKP_LATTICE_MIX_ANNEAL_EPOCHS <= 0:
        return float(GKP_LATTICE_MIX_START)
    frac = np.clip(float(epoch) / float(GKP_LATTICE_MIX_ANNEAL_EPOCHS), 0.0, 1.0)
    return float((1.0 - frac) * GKP_LATTICE_MIX_START + frac * GKP_LATTICE_MIX_END)


def _sample_characteristic_points(rng, n_points, mode=None, bank=None, lattice_mix=None):
    if bank is None:
        bank = FINAL_BANK
    char_points = bank["char_points"]
    char_target = bank["char_target"]
    char_weights = bank["char_weights"]
    char_radii = bank["char_radii"]
    mix = float(GKP_LATTICE_MIX_START if lattice_mix is None else lattice_mix)
    mix = float(np.clip(mix, 0.0, 1.0))

    mode = CHAR_SAMPLER_MODE if mode is None else mode

    def _apply_lattice_mix(idx, q_base):
        q_base = np.asarray(q_base, dtype=float)
        q_base = np.maximum(q_base, 1e-12)
        q_base = q_base / float(np.sum(q_base))
        idx = np.asarray(idx, dtype=int)
        lattice_probs = np.asarray(bank.get("lattice_probs", q_base), dtype=float)
        if lattice_probs.shape != q_base.shape:
            lattice_probs = q_base
        lattice_probs = np.maximum(lattice_probs, 1e-12)
        lattice_probs = lattice_probs / float(np.sum(lattice_probs))
        if mix <= 0.0:
            return idx, q_base[idx]
        n_lattice = int(round(mix * n_points))
        if n_lattice > 0:
            repl_pos = rng.choice(n_points, size=n_lattice, replace=False)
            repl_idx = rng.choice(len(q_base), size=n_lattice, replace=True, p=lattice_probs)
            idx = idx.copy()
            idx[repl_pos] = repl_idx
        q_mix = (1.0 - mix) * q_base + mix * lattice_probs
        q_mix = np.maximum(q_mix, 1e-12)
        q_mix = q_mix / float(np.sum(q_mix))
        return idx, q_mix[idx]

    if mode == "weighted":
        idx = rng.choice(len(char_points), size=n_points, replace=True, p=char_weights)
        idx, samp_probs = _apply_lattice_mix(idx, char_weights)
    elif mode == "uniform":
        uniform_q = np.full(len(char_points), 1.0 / len(char_points), dtype=float)
        idx = rng.choice(len(char_points), size=n_points, replace=True, p=uniform_q)
        idx, samp_probs = _apply_lattice_mix(idx, uniform_q)
    elif mode == "radial_stratified":
        n_bins = max(1, CHAR_RADIAL_BINS)
        r_max = float(np.max(char_radii))
        edges = np.linspace(0.0, r_max + 1e-12, n_bins + 1)
        idx_list = []
        bin_candidates = []
        bin_mass = np.zeros(n_bins, dtype=float)
        for bi in range(n_bins):
            lo = edges[bi]
            hi = edges[bi + 1]
            if bi == n_bins - 1:
                mask = (char_radii >= lo) & (char_radii <= hi)
            else:
                mask = (char_radii >= lo) & (char_radii < hi)
            candidates = np.flatnonzero(mask)
            bin_candidates.append(candidates)
            if candidates.size > 0:
                bin_mass[bi] = float(np.sum(char_weights[candidates]))

        mass_total = float(np.sum(bin_mass))
        if mass_total <= 0.0 or not np.isfinite(mass_total):
            quotas = np.full(n_bins, n_points // n_bins, dtype=int)
            quotas[: (n_points % n_bins)] += 1
        else:
            raw = n_points * (bin_mass / mass_total)
            quotas = np.floor(raw).astype(int)
            remaining = int(n_points - int(np.sum(quotas)))
            if remaining > 0:
                frac = raw - quotas
                order = np.argsort(frac)[::-1]
                for bi in order[:remaining]:
                    quotas[bi] += 1
        # q(alpha): actual sampling distribution induced by stratified sampling.
        # This must be used for importance weighting in reward evaluation.
        q = np.zeros(len(char_points), dtype=float)
        for bi in range(n_bins):
            candidates = bin_candidates[bi]
            if candidates.size == 0:
                continue
            take = min(int(quotas[bi]), n_points - len(idx_list))
            if take <= 0:
                break
            local_w = char_weights[candidates]
            local_w_sum = float(np.sum(local_w))
            if local_w_sum > 0.0 and np.isfinite(local_w_sum):
                local_w = local_w / local_w_sum
                sampled = rng.choice(candidates, size=take, replace=True, p=local_w)
                q[candidates] += (take / float(n_points)) * local_w
            else:
                sampled = rng.choice(candidates, size=take, replace=True)
                q[candidates] += (take / float(n_points)) / float(candidates.size)
            idx_list.extend(sampled.tolist())
        n_fill = n_points - len(idx_list)
        if n_fill > 0:
            fill = rng.choice(
                len(char_points),
                size=n_fill,
                replace=True,
                p=char_weights,
            )
            idx_list.extend(fill.tolist())
            q += (n_fill / float(n_points)) * char_weights
        idx = np.asarray(idx_list, dtype=int)
        q = np.maximum(q, 1e-12)
        q = q / float(np.sum(q))
        idx, samp_probs = _apply_lattice_mix(idx, q)
    else:
        raise ValueError(f"Unknown CHAR_SAMPLER_MODE={mode}")
    points = [char_points[i] for i in idx]
    targets = char_target[idx]
    weights = np.asarray(samp_probs, dtype=float)
    return points, targets, weights


def _select_train_points(epoch, rng, bank):
    lattice_mix = _lattice_mix_for_epoch(epoch)
    stage1_norm = bank["topk_overlap_norm"] if CHAR_USE_FIXED_REWARD_NORM else None
    stage23_norm = bank["char_overlap_norm"] if CHAR_USE_FIXED_REWARD_NORM else None
    if epoch < TRAIN_STAGE1_EPOCHS:
        return (
            bank["topk_points"],
            bank["topk_target"],
            bank["topk_weights"],
            stage1_norm,
        )
    if epoch < TRAIN_STAGE2_EPOCHS:
        points, targets, weights = _sample_characteristic_points(
            rng,
            TRAIN_POINTS_STAGE2,
            bank=bank,
            lattice_mix=lattice_mix,
        )
        return points, targets, weights, stage23_norm
    points, targets, weights = _sample_characteristic_points(
        rng,
        TRAIN_POINTS_STAGE3,
        bank=bank,
        lattice_mix=lattice_mix,
    )
    return points, targets, weights, stage23_norm


EVAL_RNG = np.random.default_rng(12345)
EVAL_LATTICE_MIX = (
    GKP_LATTICE_MIX_END if GKP_LATTICE_MIX_ANNEAL_EPOCHS > 0 else GKP_LATTICE_MIX_START
)
EVAL_POINTS, EVAL_TARGET, EVAL_WEIGHTS = _sample_characteristic_points(
    EVAL_RNG,
    TRAIN_POINTS_STAGE3,
    mode=CHAR_SAMPLER_MODE,
    bank=FINAL_BANK,
    lattice_mix=EVAL_LATTICE_MIX,
)
if CHAR_USE_FIXED_REWARD_NORM:
    _eval_q = np.maximum(np.asarray(EVAL_WEIGHTS, dtype=float), 1.0e-12)
    EVAL_REWARD_NORM = float(
        np.mean((np.abs(np.asarray(EVAL_TARGET, dtype=complex)) ** 2) / _eval_q)
    )
    if not np.isfinite(EVAL_REWARD_NORM) or EVAL_REWARD_NORM <= 0.0:
        EVAL_REWARD_NORM = FINAL_OVERLAP_NORM
else:
    EVAL_REWARD_NORM = None


def _eval_fidelity_batch_full(phi_r_full, phi_b_full, amp_r_full, amp_b_full):
    _, fidelity_batch, _, _ = trapped_ion_gkp_sim_batch(
        np.asarray(phi_r_full, dtype=float),
        np.asarray(phi_b_full, dtype=float),
        amp_r=amp_r_full,
        amp_b=amp_b_full,
        n_boson=N_BOSON,
        omega=2 * np.pi * 0.002,
        t_step=T_STEP,
        alpha_cat=GKP_DELTA,
        cat_parity=GKP_LOGICAL,
        gkp_delta=GKP_DELTA,
        gkp_logical=GKP_LOGICAL,
        cat_phase=GKP_REL_PHASE,
        sample_points=[0.0 + 0.0j],
        target_values=np.array([1.0 + 0.0j], dtype=complex),
        sample_weights=np.array([1.0], dtype=float),
        sample_area=1.0,
        reward_scale=1.0,
        reward_clip=None,
        reward_norm=None,
        n_shots=0,
        return_details=True,
        return_density=False,
        reward_mode="characteristic",
        characteristic_objective=CHAR_REWARD_OBJECTIVE_STAGE2,
        gkp_squeeze_r=GKP_SQUEEZE_R,
        gkp_envelope_kappa=GKP_ENVELOPE_KAPPA,
        gkp_lattice_trunc=GKP_LATTICE_TRUNC,
    )
    return np.asarray(fidelity_batch, dtype=float)


def _expand_segments(arr):
    return np.repeat(np.asarray(arr, dtype=float), SEG_LEN, axis=1)


def _phase_weighted_mean(samples, weights):
    z = np.sum(np.exp(1j * samples) * weights[:, None], axis=0)
    return np.angle(z)


def _phase_weighted_std(samples, center, weights):
    delta = np.angle(np.exp(1j * (samples - center[None, :])))
    var = np.sum(weights[:, None] * (delta ** 2), axis=0)
    return np.sqrt(np.maximum(var, 0.0))


def _weighted_mean_std(samples, weights):
    mean = np.sum(samples * weights[:, None], axis=0)
    var = np.sum(weights[:, None] * ((samples - mean[None, :]) ** 2), axis=0)
    return mean, np.sqrt(np.maximum(var, 0.0))


def _elite_weights(fidelity_vals):
    vals = np.asarray(fidelity_vals, dtype=float)
    shifted = vals - float(np.min(vals))
    shifted = np.maximum(shifted, 1.0e-12)
    weights = shifted ** FINAL_REFINE_ELITE_WEIGHT_POWER
    s = float(np.sum(weights))
    if s <= 0.0 or not np.isfinite(s):
        return np.full_like(vals, 1.0 / max(1, vals.size), dtype=float)
    return weights / s


def _eval_fidelity_batch(phi_r_coeff, phi_b_coeff, amp_r_coeff, amp_b_coeff):
    return _eval_fidelity_batch_full(
        _expand_segments(phi_r_coeff),
        _expand_segments(phi_b_coeff),
        _expand_segments(amp_r_coeff),
        _expand_segments(amp_b_coeff),
    )


def _refine_full_steps_phase(
    center_phi_r,
    center_phi_b,
    center_amp_r,
    center_amp_b,
    sigma_phi_r,
    sigma_phi_b,
    sigma_amp_r,
    sigma_amp_b,
    rng,
    deadline_ts=None,
    stop_buffer_sec=0.0,
    ):
    if not FINAL_REFINE_FULL_STEPS or FINAL_REFINE_FULL_SAMPLES <= 0:
        return (
            np.repeat(np.asarray(center_phi_r, dtype=float), SEG_LEN),
            np.repeat(np.asarray(center_phi_b, dtype=float), SEG_LEN),
            np.repeat(np.asarray(center_amp_r, dtype=float), SEG_LEN),
            np.repeat(np.asarray(center_amp_b, dtype=float), SEG_LEN),
            float(
                _eval_fidelity_batch(
                    np.asarray(center_phi_r, dtype=float)[None, :],
                    np.asarray(center_phi_b, dtype=float)[None, :],
                    np.asarray(center_amp_r, dtype=float)[None, :],
                    np.asarray(center_amp_b, dtype=float)[None, :],
                )[0]
            ),
        )

    cur_phi_r = np.repeat(np.asarray(center_phi_r, dtype=float), SEG_LEN)
    cur_phi_b = np.repeat(np.asarray(center_phi_b, dtype=float), SEG_LEN)
    best_phi_r = cur_phi_r.copy()
    best_phi_b = cur_phi_b.copy()
    best_amp_r = np.repeat(np.asarray(center_amp_r, dtype=float), SEG_LEN)
    best_amp_b = np.repeat(np.asarray(center_amp_b, dtype=float), SEG_LEN)
    cur_amp_r = best_amp_r.copy()
    cur_amp_b = best_amp_b.copy()
    sigma_phi_r_full = np.maximum(
        np.repeat(np.asarray(sigma_phi_r, dtype=float), SEG_LEN) * FINAL_REFINE_FULL_SIGMA_FACTOR,
        FINAL_REFINE_FULL_MIN_SIGMA,
    )
    sigma_phi_b_full = np.maximum(
        np.repeat(np.asarray(sigma_phi_b, dtype=float), SEG_LEN) * FINAL_REFINE_FULL_SIGMA_FACTOR,
        FINAL_REFINE_FULL_MIN_SIGMA,
    )
    sigma_amp_r_full = np.maximum(
        np.repeat(np.asarray(sigma_amp_r, dtype=float), SEG_LEN)
        * FINAL_REFINE_FULL_SIGMA_FACTOR_AMP,
        FINAL_REFINE_FULL_MIN_SIGMA_AMP,
    )
    sigma_amp_b_full = np.maximum(
        np.repeat(np.asarray(sigma_amp_b, dtype=float), SEG_LEN)
        * FINAL_REFINE_FULL_SIGMA_FACTOR_AMP,
        FINAL_REFINE_FULL_MIN_SIGMA_AMP,
    )
    best_fidelity = float(
        _eval_fidelity_batch_full(
            cur_phi_r[None, :],
            cur_phi_b[None, :],
            best_amp_r[None, :],
            best_amp_b[None, :],
        )[0]
    )

    n_rounds = max(1, FINAL_REFINE_FULL_ROUNDS)
    n_samples = max(0, FINAL_REFINE_FULL_SAMPLES)
    topk = max(1, FINAL_REFINE_FULL_TOPK)
    for ridx in range(n_rounds):
        if _deadline_exhausted(deadline_ts, stop_buffer_sec):
            logger.info(
                "Final full-step refine budget stop at round %d/%d (remaining %.1fs)",
                ridx + 1,
                n_rounds,
                _deadline_remaining(deadline_ts),
            )
            break
        if n_samples <= 0:
            break
        scale = FINAL_REFINE_FULL_SCALE * (FINAL_REFINE_FULL_DECAY ** ridx)
        n_cand = n_samples + 1
        cand_phi_r = np.repeat(cur_phi_r[None, :], n_cand, axis=0)
        cand_phi_b = np.repeat(cur_phi_b[None, :], n_cand, axis=0)
        cand_amp_r = np.repeat(cur_amp_r[None, :], n_cand, axis=0)
        cand_amp_b = np.repeat(cur_amp_b[None, :], n_cand, axis=0)

        noise_r = rng.normal(size=(n_samples, N_STEPS))
        noise_b = rng.normal(size=(n_samples, N_STEPS))
        cand_phi_r[1:, :] = cur_phi_r[None, :] + scale * noise_r * sigma_phi_r_full[None, :]
        cand_phi_b[1:, :] = cur_phi_b[None, :] + scale * noise_b * sigma_phi_b_full[None, :]
        if FINAL_REFINE_FULL_ENABLE_AMP:
            noise_amp_r = rng.normal(size=(n_samples, N_STEPS))
            noise_amp_b = rng.normal(size=(n_samples, N_STEPS))
            cand_amp_r[1:, :] = (
                cur_amp_r[None, :] + scale * noise_amp_r * sigma_amp_r_full[None, :]
            )
            cand_amp_b[1:, :] = (
                cur_amp_b[None, :] + scale * noise_amp_b * sigma_amp_b_full[None, :]
            )
        cand_phi_r = _project_phase(cand_phi_r)
        cand_phi_b = _project_phase(cand_phi_b)
        cand_amp_r = np.clip(cand_amp_r, AMP_MIN, AMP_MAX)
        cand_amp_b = np.clip(cand_amp_b, AMP_MIN, AMP_MAX)

        cand_fidelity = _eval_fidelity_batch_full(cand_phi_r, cand_phi_b, cand_amp_r, cand_amp_b)
        order = np.argsort(cand_fidelity)[::-1]
        keep = order[: min(topk, len(order))]
        elite_count = max(1, int(np.ceil(FINAL_REFINE_FULL_ELITE_FRACTION * len(keep))))
        elite = keep[:elite_count]
        round_best_idx = int(order[0])
        round_best = float(cand_fidelity[round_best_idx])
        improved = round_best > best_fidelity
        logger.info(
            "Final full-step refine round %d/%d | scale=%.4f best=%.6f mean_topk=%.6f elite=%d",
            ridx + 1,
            n_rounds,
            scale,
            round_best,
            float(np.mean(cand_fidelity[keep])),
            elite_count,
        )

        if improved:
            best_fidelity = round_best
            best_phi_r = cand_phi_r[round_best_idx].copy()
            best_phi_b = cand_phi_b[round_best_idx].copy()
            best_amp_r = cand_amp_r[round_best_idx].copy()
            best_amp_b = cand_amp_b[round_best_idx].copy()

        if FINAL_REFINE_FULL_UPDATE_MODE == "elite_mean":
            elite_w = _elite_weights(cand_fidelity[elite])
            cur_phi_r = _project_phase(_phase_weighted_mean(cand_phi_r[elite], elite_w))
            cur_phi_b = _project_phase(_phase_weighted_mean(cand_phi_b[elite], elite_w))
            sigma_phi_r_full = np.maximum(
                _phase_weighted_std(cand_phi_r[elite], cur_phi_r, elite_w),
                FINAL_REFINE_FULL_MIN_SIGMA,
            )
            sigma_phi_b_full = np.maximum(
                _phase_weighted_std(cand_phi_b[elite], cur_phi_b, elite_w),
                FINAL_REFINE_FULL_MIN_SIGMA,
            )
            if FINAL_REFINE_FULL_ENABLE_AMP:
                cur_amp_r, sigma_amp_r_full = _weighted_mean_std(cand_amp_r[elite], elite_w)
                cur_amp_b, sigma_amp_b_full = _weighted_mean_std(cand_amp_b[elite], elite_w)
                cur_amp_r = np.clip(cur_amp_r, AMP_MIN, AMP_MAX)
                cur_amp_b = np.clip(cur_amp_b, AMP_MIN, AMP_MAX)
                sigma_amp_r_full = np.maximum(
                    sigma_amp_r_full, FINAL_REFINE_FULL_MIN_SIGMA_AMP
                )
                sigma_amp_b_full = np.maximum(
                    sigma_amp_b_full, FINAL_REFINE_FULL_MIN_SIGMA_AMP
                )
        else:
            if improved:
                cur_phi_r = best_phi_r.copy()
                cur_phi_b = best_phi_b.copy()
                if FINAL_REFINE_FULL_ENABLE_AMP:
                    cur_amp_r = best_amp_r.copy()
                    cur_amp_b = best_amp_b.copy()
            sigma_phi_r_full = np.maximum(
                np.std(cand_phi_r[keep], axis=0), FINAL_REFINE_FULL_MIN_SIGMA
            )
            sigma_phi_b_full = np.maximum(
                np.std(cand_phi_b[keep], axis=0), FINAL_REFINE_FULL_MIN_SIGMA
            )
            if FINAL_REFINE_FULL_ENABLE_AMP:
                sigma_amp_r_full = np.maximum(
                    np.std(cand_amp_r[keep], axis=0), FINAL_REFINE_FULL_MIN_SIGMA_AMP
                )
                sigma_amp_b_full = np.maximum(
                    np.std(cand_amp_b[keep], axis=0), FINAL_REFINE_FULL_MIN_SIGMA_AMP
                )

    return best_phi_r, best_phi_b, best_amp_r, best_amp_b, best_fidelity


def _spsa_polish_full_steps(
    center_phi_r,
    center_phi_b,
    center_amp_r,
    center_amp_b,
    rng,
    num_iters=None,
    patience=None,
    log_prefix="Final SPSA polish",
    allow_auto_extend=None,
    deadline_ts=None,
    stop_buffer_sec=0.0,
):
    best_phi_r = np.asarray(center_phi_r, dtype=float).copy()
    best_phi_b = np.asarray(center_phi_b, dtype=float).copy()
    best_amp_r = np.asarray(center_amp_r, dtype=float).copy()
    best_amp_b = np.asarray(center_amp_b, dtype=float).copy()
    best_fidelity = float(
        _eval_fidelity_batch_full(
            best_phi_r[None, :],
            best_phi_b[None, :],
            best_amp_r[None, :],
            best_amp_b[None, :],
        )[0]
    )
    if not FINAL_POLISH_ENABLE:
        return best_phi_r, best_phi_b, best_amp_r, best_amp_b, best_fidelity

    n_iters = FINAL_POLISH_ITERS if num_iters is None else int(num_iters)
    if n_iters <= 0:
        return best_phi_r, best_phi_b, best_amp_r, best_amp_b, best_fidelity
    cur_phi_r = best_phi_r.copy()
    cur_phi_b = best_phi_b.copy()
    cur_amp_r = best_amp_r.copy()
    cur_amp_b = best_amp_b.copy()
    cur_fidelity = best_fidelity
    patience_limit = FINAL_POLISH_PATIENCE if patience is None else int(patience)
    max_iters = max(n_iters, FINAL_POLISH_MAX_ITERS)
    if allow_auto_extend is None:
        allow_auto_extend = FINAL_POLISH_AUTO_EXTEND and (num_iters is None)
    target_iters = n_iters
    block_start_best = best_fidelity

    no_improve = 0
    k = 0
    while k < target_iters:
        if _deadline_exhausted(deadline_ts, stop_buffer_sec):
            logger.info(
                "%s budget stop at iter %d/%d (remaining %.1fs)",
                log_prefix,
                k + 1,
                target_iters,
                _deadline_remaining(deadline_ts),
            )
            break
        step_idx = float(k + 1)
        ak_phase = FINAL_POLISH_PHASE_A0 / (step_idx ** FINAL_POLISH_ALPHA)
        ck_phase = FINAL_POLISH_PHASE_C0 / (step_idx ** FINAL_POLISH_GAMMA)
        ak_amp = FINAL_POLISH_AMP_A0 / (step_idx ** FINAL_POLISH_ALPHA)
        ck_amp = FINAL_POLISH_AMP_C0 / (step_idx ** FINAL_POLISH_GAMMA)

        dphi_r = rng.choice([-1.0, 1.0], size=N_STEPS)
        dphi_b = rng.choice([-1.0, 1.0], size=N_STEPS)

        phi_r_plus = cur_phi_r + ck_phase * dphi_r
        phi_r_minus = cur_phi_r - ck_phase * dphi_r
        phi_b_plus = cur_phi_b + ck_phase * dphi_b
        phi_b_minus = cur_phi_b - ck_phase * dphi_b
        amp_r_plus = cur_amp_r.copy()
        amp_r_minus = cur_amp_r.copy()
        amp_b_plus = cur_amp_b.copy()
        amp_b_minus = cur_amp_b.copy()
        if FINAL_POLISH_INCLUDE_AMP:
            damp_r = rng.choice([-1.0, 1.0], size=N_STEPS)
            damp_b = rng.choice([-1.0, 1.0], size=N_STEPS)
            amp_r_plus = cur_amp_r + ck_amp * damp_r
            amp_r_minus = cur_amp_r - ck_amp * damp_r
            amp_b_plus = cur_amp_b + ck_amp * damp_b
            amp_b_minus = cur_amp_b - ck_amp * damp_b

        phi_r_plus = _project_phase(phi_r_plus)
        phi_r_minus = _project_phase(phi_r_minus)
        phi_b_plus = _project_phase(phi_b_plus)
        phi_b_minus = _project_phase(phi_b_minus)
        amp_r_plus = np.clip(amp_r_plus, AMP_MIN, AMP_MAX)
        amp_r_minus = np.clip(amp_r_minus, AMP_MIN, AMP_MAX)
        amp_b_plus = np.clip(amp_b_plus, AMP_MIN, AMP_MAX)
        amp_b_minus = np.clip(amp_b_minus, AMP_MIN, AMP_MAX)

        two_fid = _eval_fidelity_batch_full(
            np.stack([phi_r_plus, phi_r_minus], axis=0),
            np.stack([phi_b_plus, phi_b_minus], axis=0),
            np.stack([amp_r_plus, amp_r_minus], axis=0),
            np.stack([amp_b_plus, amp_b_minus], axis=0),
        )
        f_plus = float(two_fid[0])
        f_minus = float(two_fid[1])
        diff = f_plus - f_minus

        g_scale_phase = diff / max(2.0 * ck_phase, 1e-12)
        cand_phi_r = cur_phi_r + ak_phase * g_scale_phase * dphi_r
        cand_phi_b = cur_phi_b + ak_phase * g_scale_phase * dphi_b
        cand_amp_r = cur_amp_r.copy()
        cand_amp_b = cur_amp_b.copy()
        if FINAL_POLISH_INCLUDE_AMP:
            g_scale_amp = diff / max(2.0 * ck_amp, 1e-12)
            cand_amp_r = cur_amp_r + ak_amp * g_scale_amp * damp_r
            cand_amp_b = cur_amp_b + ak_amp * g_scale_amp * damp_b
        cand_phi_r = _project_phase(cand_phi_r)
        cand_phi_b = _project_phase(cand_phi_b)
        cand_amp_r = np.clip(cand_amp_r, AMP_MIN, AMP_MAX)
        cand_amp_b = np.clip(cand_amp_b, AMP_MIN, AMP_MAX)
        f_cand = float(
            _eval_fidelity_batch_full(
                cand_phi_r[None, :],
                cand_phi_b[None, :],
                cand_amp_r[None, :],
                cand_amp_b[None, :],
            )[0]
        )

        if FINAL_POLISH_ACCEPT_MODE == "spsa":
            # True SPSA-style trajectory update: follow the gradient step, and
            # track the best point separately for return.
            cur_phi_r = cand_phi_r.copy()
            cur_phi_b = cand_phi_b.copy()
            cur_amp_r = cand_amp_r.copy()
            cur_amp_b = cand_amp_b.copy()
            cur_fidelity = f_cand
        else:
            # Greedy hill-climb mode (historical behavior).
            cur_iter_fid = cur_fidelity
            cur_iter_state = None
            if f_plus > cur_iter_fid:
                cur_iter_fid = f_plus
                cur_iter_state = (phi_r_plus, phi_b_plus, amp_r_plus, amp_b_plus)
            if f_minus > cur_iter_fid:
                cur_iter_fid = f_minus
                cur_iter_state = (phi_r_minus, phi_b_minus, amp_r_minus, amp_b_minus)
            if f_cand > cur_iter_fid:
                cur_iter_fid = f_cand
                cur_iter_state = (cand_phi_r, cand_phi_b, cand_amp_r, cand_amp_b)
            if cur_iter_state is not None:
                cur_phi_r, cur_phi_b, cur_amp_r, cur_amp_b = cur_iter_state
                cur_fidelity = cur_iter_fid

        best_iter_fid = best_fidelity
        best_iter_state = None
        for fid_val, state_val in (
            (f_plus, (phi_r_plus, phi_b_plus, amp_r_plus, amp_b_plus)),
            (f_minus, (phi_r_minus, phi_b_minus, amp_r_minus, amp_b_minus)),
            (f_cand, (cand_phi_r, cand_phi_b, cand_amp_r, cand_amp_b)),
        ):
            if fid_val > best_iter_fid:
                best_iter_fid = fid_val
                best_iter_state = state_val
        if best_iter_state is not None:
            best_phi_r, best_phi_b, best_amp_r, best_amp_b = best_iter_state
            best_fidelity = best_iter_fid
            no_improve = 0
        else:
            no_improve += 1

        if (k + 1) % FINAL_POLISH_LOG_INTERVAL == 0 or k == 0:
            logger.info(
                "%s iter %d/%d | f+=%.6f f-=%.6f f_cand=%.6f cur=%.6f best=%.6f",
                log_prefix,
                k + 1,
                target_iters,
                f_plus,
                f_minus,
                f_cand,
                cur_fidelity,
                best_fidelity,
            )

        if patience_limit > 0 and no_improve >= patience_limit:
            logger.info(
                "%s early stop at iter %d (no improvement for %d iters)",
                log_prefix,
                k + 1,
                no_improve,
            )
            break

        k += 1
        # Extend budget only if the recent block still produces meaningful gains.
        if (
            allow_auto_extend
            and k > 0
            and (k % FINAL_POLISH_EXTEND_BLOCK == 0 or k == target_iters)
            and not _deadline_exhausted(deadline_ts, stop_buffer_sec + 15.0)
            and target_iters < max_iters
        ):
            block_gain = best_fidelity - block_start_best
            block_start_best = best_fidelity
            if block_gain >= FINAL_POLISH_EXTEND_MIN_GAIN:
                new_target = min(target_iters + FINAL_POLISH_EXTEND_BLOCK, max_iters)
                if new_target > target_iters:
                    logger.info(
                        "%s auto-extend: block_gain=%.6f -> extending budget %d -> %d (max=%d)",
                        log_prefix,
                        block_gain,
                        target_iters,
                        new_target,
                        max_iters,
                    )
                    target_iters = new_target

    return best_phi_r, best_phi_b, best_amp_r, best_amp_b, best_fidelity


def _jitter_full_steps_state(
    phi_r,
    phi_b,
    amp_r,
    amp_b,
    phase_sigma,
    amp_sigma,
    rng,
):
    out_phi_r = np.asarray(phi_r, dtype=float).copy()
    out_phi_b = np.asarray(phi_b, dtype=float).copy()
    out_amp_r = np.asarray(amp_r, dtype=float).copy()
    out_amp_b = np.asarray(amp_b, dtype=float).copy()
    if phase_sigma > 0.0:
        out_phi_r = out_phi_r + rng.normal(0.0, phase_sigma, size=out_phi_r.shape)
        out_phi_b = out_phi_b + rng.normal(0.0, phase_sigma, size=out_phi_b.shape)
    if amp_sigma > 0.0:
        out_amp_r = out_amp_r + rng.normal(0.0, amp_sigma, size=out_amp_r.shape)
        out_amp_b = out_amp_b + rng.normal(0.0, amp_sigma, size=out_amp_b.shape)
    out_phi_r = _project_phase(out_phi_r)
    out_phi_b = _project_phase(out_phi_b)
    out_amp_r = np.clip(out_amp_r, AMP_MIN, AMP_MAX)
    out_amp_b = np.clip(out_amp_b, AMP_MIN, AMP_MAX)
    return out_phi_r, out_phi_b, out_amp_r, out_amp_b


def _coordinate_polish_full_steps(
    center_phi_r,
    center_phi_b,
    center_amp_r,
    center_amp_b,
    deadline_ts=None,
    stop_buffer_sec=0.0,
):
    cur_phi_r = np.asarray(center_phi_r, dtype=float).copy()
    cur_phi_b = np.asarray(center_phi_b, dtype=float).copy()
    cur_amp_r = np.asarray(center_amp_r, dtype=float).copy()
    cur_amp_b = np.asarray(center_amp_b, dtype=float).copy()
    best_fidelity = float(
        _eval_fidelity_batch_full(
            cur_phi_r[None, :],
            cur_phi_b[None, :],
            cur_amp_r[None, :],
            cur_amp_b[None, :],
        )[0]
    )
    if not FINAL_COORD_POLISH_ENABLE or FINAL_COORD_POLISH_SWEEPS <= 0:
        return cur_phi_r, cur_phi_b, cur_amp_r, cur_amp_b, best_fidelity

    phase_step = float(FINAL_COORD_POLISH_PHASE_STEP)
    amp_step = float(FINAL_COORD_POLISH_AMP_STEP)
    include_amp = FINAL_COORD_POLISH_INCLUDE_AMP
    batch_size = max(1, int(FINAL_COORD_POLISH_BATCH_SIZE))
    empty_sweep_streak = 0

    for sweep in range(FINAL_COORD_POLISH_SWEEPS):
        if _deadline_exhausted(deadline_ts, stop_buffer_sec):
            logger.info(
                "Coordinate polish budget stop before sweep %d/%d (remaining %.1fs)",
                sweep + 1,
                FINAL_COORD_POLISH_SWEEPS,
                _deadline_remaining(deadline_ts),
            )
            break
        sweep_start = best_fidelity
        moves_taken = 0

        for move in range(FINAL_COORD_POLISH_MAX_MOVES):
            if _deadline_exhausted(deadline_ts, stop_buffer_sec):
                logger.info(
                    "Coordinate polish budget stop at sweep %d/%d move %d/%d (remaining %.1fs)",
                    sweep + 1,
                    FINAL_COORD_POLISH_SWEEPS,
                    move + 1,
                    FINAL_COORD_POLISH_MAX_MOVES,
                    _deadline_remaining(deadline_ts),
                )
                break
            cand_phi_r = []
            cand_phi_b = []
            cand_amp_r = []
            cand_amp_b = []

            if phase_step >= FINAL_COORD_POLISH_PHASE_MIN_STEP:
                for idx in range(N_STEPS):
                    for sign in (1.0, -1.0):
                        cpr = cur_phi_r.copy()
                        cpr[idx] = _project_phase_scalar(cpr[idx] + sign * phase_step)
                        cand_phi_r.append(cpr)
                        cand_phi_b.append(cur_phi_b.copy())
                        cand_amp_r.append(cur_amp_r.copy())
                        cand_amp_b.append(cur_amp_b.copy())
                for idx in range(N_STEPS):
                    for sign in (1.0, -1.0):
                        cpb = cur_phi_b.copy()
                        cpb[idx] = _project_phase_scalar(cpb[idx] + sign * phase_step)
                        cand_phi_r.append(cur_phi_r.copy())
                        cand_phi_b.append(cpb)
                        cand_amp_r.append(cur_amp_r.copy())
                        cand_amp_b.append(cur_amp_b.copy())

            if include_amp and amp_step >= FINAL_COORD_POLISH_AMP_MIN_STEP:
                for idx in range(N_STEPS):
                    for sign in (1.0, -1.0):
                        car = cur_amp_r.copy()
                        car[idx] = np.clip(car[idx] + sign * amp_step, AMP_MIN, AMP_MAX)
                        cand_phi_r.append(cur_phi_r.copy())
                        cand_phi_b.append(cur_phi_b.copy())
                        cand_amp_r.append(car)
                        cand_amp_b.append(cur_amp_b.copy())
                for idx in range(N_STEPS):
                    for sign in (1.0, -1.0):
                        cab = cur_amp_b.copy()
                        cab[idx] = np.clip(cab[idx] + sign * amp_step, AMP_MIN, AMP_MAX)
                        cand_phi_r.append(cur_phi_r.copy())
                        cand_phi_b.append(cur_phi_b.copy())
                        cand_amp_r.append(cur_amp_r.copy())
                        cand_amp_b.append(cab)

            if not cand_phi_r:
                break

            cand_phi_r = np.asarray(cand_phi_r, dtype=float)
            cand_phi_b = np.asarray(cand_phi_b, dtype=float)
            cand_amp_r = np.asarray(cand_amp_r, dtype=float)
            cand_amp_b = np.asarray(cand_amp_b, dtype=float)

            best_move_fid = best_fidelity
            best_move_idx = -1
            n_cand = cand_phi_r.shape[0]
            for start in range(0, n_cand, batch_size):
                stop = min(start + batch_size, n_cand)
                fids = _eval_fidelity_batch_full(
                    cand_phi_r[start:stop],
                    cand_phi_b[start:stop],
                    cand_amp_r[start:stop],
                    cand_amp_b[start:stop],
                )
                local_idx = int(np.argmax(fids))
                local_best = float(fids[local_idx])
                if local_best > best_move_fid:
                    best_move_fid = local_best
                    best_move_idx = start + local_idx

            if best_move_idx < 0:
                break

            cur_phi_r = cand_phi_r[best_move_idx].copy()
            cur_phi_b = cand_phi_b[best_move_idx].copy()
            cur_amp_r = cand_amp_r[best_move_idx].copy()
            cur_amp_b = cand_amp_b[best_move_idx].copy()
            best_fidelity = best_move_fid
            moves_taken += 1

            logger.info(
                "Coordinate polish sweep %d/%d move %d/%d | best=%.6f",
                sweep + 1,
                FINAL_COORD_POLISH_SWEEPS,
                move + 1,
                FINAL_COORD_POLISH_MAX_MOVES,
                best_fidelity,
            )

        sweep_gain = best_fidelity - sweep_start
        if moves_taken == 0:
            empty_sweep_streak += 1
        else:
            empty_sweep_streak = 0
        logger.info(
            "Coordinate polish sweep %d/%d done | gain=%.6f moves=%d phase_step=%.6g amp_step=%.6g best=%.6f",
            sweep + 1,
            FINAL_COORD_POLISH_SWEEPS,
            sweep_gain,
            moves_taken,
            phase_step,
            amp_step if include_amp else 0.0,
            best_fidelity,
        )
        next_phase_step = max(
            phase_step * FINAL_COORD_POLISH_PHASE_DECAY,
            FINAL_COORD_POLISH_PHASE_MIN_STEP,
        )
        next_amp_step = amp_step
        if include_amp:
            next_amp_step = max(
                amp_step * FINAL_COORD_POLISH_AMP_DECAY,
                FINAL_COORD_POLISH_AMP_MIN_STEP,
            )
        can_refine_finer = (
            (phase_step > FINAL_COORD_POLISH_PHASE_MIN_STEP + 1.0e-15)
            or (
                include_amp
                and amp_step > FINAL_COORD_POLISH_AMP_MIN_STEP + 1.0e-15
            )
        )
        if sweep_gain < FINAL_COORD_POLISH_MIN_GAIN:
            if (
                can_refine_finer
                and empty_sweep_streak <= FINAL_COORD_POLISH_EMPTY_SWEEP_PATIENCE
            ):
                logger.info(
                    "Coordinate polish low-gain continuation: gain %.6f < %.6f, trying finer steps (empty_sweep_streak=%d/%d)",
                    sweep_gain,
                    FINAL_COORD_POLISH_MIN_GAIN,
                    empty_sweep_streak,
                    FINAL_COORD_POLISH_EMPTY_SWEEP_PATIENCE,
                )
            else:
                break

        phase_step = next_phase_step
        if include_amp:
            amp_step = next_amp_step

    return cur_phi_r, cur_phi_b, cur_amp_r, cur_amp_b, best_fidelity


def _refine_around_center(
    center_phi_r,
    center_phi_b,
    center_amp_r,
    center_amp_b,
    sigma_phi_r,
    sigma_phi_b,
    sigma_amp_r,
    sigma_amp_b,
    rng,
    deadline_ts=None,
    stop_buffer_sec=0.0,
):
    center_phi_r = np.asarray(center_phi_r, dtype=float)
    center_phi_b = np.asarray(center_phi_b, dtype=float)
    center_amp_r = np.asarray(center_amp_r, dtype=float)
    center_amp_b = np.asarray(center_amp_b, dtype=float)
    sigma_phi_r = np.asarray(sigma_phi_r, dtype=float)
    sigma_phi_b = np.asarray(sigma_phi_b, dtype=float)
    sigma_amp_r = np.asarray(sigma_amp_r, dtype=float)
    sigma_amp_b = np.asarray(sigma_amp_b, dtype=float)

    cur_phi_r = center_phi_r.copy()
    cur_phi_b = center_phi_b.copy()
    cur_amp_r = center_amp_r.copy()
    cur_amp_b = center_amp_b.copy()
    best_phi_r = cur_phi_r.copy()
    best_phi_b = cur_phi_b.copy()
    best_amp_r = cur_amp_r.copy()
    best_amp_b = cur_amp_b.copy()
    best_fidelity = float(
        _eval_fidelity_batch(
            cur_phi_r[None, :],
            cur_phi_b[None, :],
            cur_amp_r[None, :],
            cur_amp_b[None, :],
        )[0]
    )

    n_rounds = max(1, FINAL_REFINE_ROUNDS)
    n_samples = max(0, FINAL_REFINE_SAMPLES)
    topk = max(1, FINAL_REFINE_TOPK)

    for ridx in range(n_rounds):
        if _deadline_exhausted(deadline_ts, stop_buffer_sec):
            logger.info(
                "Final segment refine budget stop at round %d/%d (remaining %.1fs)",
                ridx + 1,
                n_rounds,
                _deadline_remaining(deadline_ts),
            )
            break
        if n_samples <= 0:
            break
        scale = FINAL_REFINE_SCALE * (FINAL_REFINE_DECAY ** ridx)
        n_cand = n_samples + 1
        amp_opt_active = FINAL_REFINE_ENABLE_AMP and (ridx >= FINAL_REFINE_AMP_START_ROUND)

        cand_phi_r = np.repeat(cur_phi_r[None, :], n_cand, axis=0)
        cand_phi_b = np.repeat(cur_phi_b[None, :], n_cand, axis=0)
        cand_amp_r = np.repeat(cur_amp_r[None, :], n_cand, axis=0)
        cand_amp_b = np.repeat(cur_amp_b[None, :], n_cand, axis=0)

        noise_r = rng.normal(size=(n_samples, N_SEGMENTS))
        noise_b = rng.normal(size=(n_samples, N_SEGMENTS))
        cand_phi_r[1:, :] = cur_phi_r[None, :] + scale * noise_r * sigma_phi_r[None, :]
        cand_phi_b[1:, :] = cur_phi_b[None, :] + scale * noise_b * sigma_phi_b[None, :]
        if amp_opt_active:
            noise_amp_r = rng.normal(size=(n_samples, N_SEGMENTS))
            noise_amp_b = rng.normal(size=(n_samples, N_SEGMENTS))
            cand_amp_r[1:, :] = (
                cur_amp_r[None, :] + scale * noise_amp_r * sigma_amp_r[None, :]
            )
            cand_amp_b[1:, :] = (
                cur_amp_b[None, :] + scale * noise_amp_b * sigma_amp_b[None, :]
            )
        cand_phi_r = _project_phase(cand_phi_r)
        cand_phi_b = _project_phase(cand_phi_b)
        cand_amp_r = np.clip(cand_amp_r, AMP_MIN, AMP_MAX)
        cand_amp_b = np.clip(cand_amp_b, AMP_MIN, AMP_MAX)

        cand_fidelity = _eval_fidelity_batch(cand_phi_r, cand_phi_b, cand_amp_r, cand_amp_b)
        order = np.argsort(cand_fidelity)[::-1]
        keep = order[: min(topk, len(order))]
        elite_count = max(1, int(np.ceil(FINAL_REFINE_SEG_ELITE_FRACTION * len(keep))))
        elite = keep[:elite_count]
        round_best_idx = int(order[0])
        round_best = float(cand_fidelity[round_best_idx])
        improved = round_best > best_fidelity
        logger.info(
            "Final refine round %d/%d | scale=%.4f best=%.6f mean_topk=%.6f elite=%d",
            ridx + 1,
            n_rounds,
            scale,
            round_best,
            float(np.mean(cand_fidelity[keep])),
            elite_count,
        )

        if improved:
            best_fidelity = round_best
            best_phi_r = cand_phi_r[round_best_idx].copy()
            best_phi_b = cand_phi_b[round_best_idx].copy()
            best_amp_r = cand_amp_r[round_best_idx].copy()
            best_amp_b = cand_amp_b[round_best_idx].copy()

        if FINAL_REFINE_SEG_UPDATE_MODE == "elite_mean":
            elite_w = _elite_weights(cand_fidelity[elite])
            cur_phi_r = _project_phase(_phase_weighted_mean(cand_phi_r[elite], elite_w))
            cur_phi_b = _project_phase(_phase_weighted_mean(cand_phi_b[elite], elite_w))
            sigma_phi_r = np.maximum(
                _phase_weighted_std(cand_phi_r[elite], cur_phi_r, elite_w),
                FINAL_REFINE_MIN_SIGMA,
            )
            sigma_phi_b = np.maximum(
                _phase_weighted_std(cand_phi_b[elite], cur_phi_b, elite_w),
                FINAL_REFINE_MIN_SIGMA,
            )
            if amp_opt_active:
                cur_amp_r, sigma_amp_r = _weighted_mean_std(cand_amp_r[elite], elite_w)
                cur_amp_b, sigma_amp_b = _weighted_mean_std(cand_amp_b[elite], elite_w)
                cur_amp_r = np.clip(cur_amp_r, AMP_MIN, AMP_MAX)
                cur_amp_b = np.clip(cur_amp_b, AMP_MIN, AMP_MAX)
                sigma_amp_r = np.maximum(sigma_amp_r, FINAL_REFINE_MIN_SIGMA_AMP)
                sigma_amp_b = np.maximum(sigma_amp_b, FINAL_REFINE_MIN_SIGMA_AMP)
        else:
            if improved:
                cur_phi_r = best_phi_r.copy()
                cur_phi_b = best_phi_b.copy()
                cur_amp_r = best_amp_r.copy()
                cur_amp_b = best_amp_b.copy()
            sigma_phi_r = np.maximum(np.std(cand_phi_r[keep], axis=0), FINAL_REFINE_MIN_SIGMA)
            sigma_phi_b = np.maximum(np.std(cand_phi_b[keep], axis=0), FINAL_REFINE_MIN_SIGMA)
            if amp_opt_active:
                sigma_amp_r = np.maximum(
                    np.std(cand_amp_r[keep], axis=0), FINAL_REFINE_MIN_SIGMA_AMP
                )
                sigma_amp_b = np.maximum(
                    np.std(cand_amp_b[keep], axis=0), FINAL_REFINE_MIN_SIGMA_AMP
                )

    return best_phi_r, best_phi_b, best_amp_r, best_amp_b, best_fidelity


def _update_top_eval_actions(
    records,
    epoch,
    fidelity,
    phi_r,
    phi_b,
    amp_r,
    amp_b,
):
    if FINAL_REFINE_TOP_EVAL_CENTERS <= 0:
        return records
    rec = {
        "epoch": int(epoch),
        "fidelity": float(fidelity),
        "phi_r": np.array(phi_r, dtype=float).copy(),
        "phi_b": np.array(phi_b, dtype=float).copy(),
        "amp_r": np.array(amp_r, dtype=float).copy(),
        "amp_b": np.array(amp_b, dtype=float).copy(),
    }
    filtered = [r for r in records if int(r["epoch"]) != int(epoch)]
    filtered.append(rec)
    filtered.sort(key=lambda x: float(x["fidelity"]), reverse=True)
    return filtered[: max(1, FINAL_REFINE_TOP_EVAL_CENTERS)]


def _update_top_train_fidelity_actions(
    records,
    epoch,
    fidelity,
    phi_r,
    phi_b,
    amp_r,
    amp_b,
):
    if TRAIN_FID_SCREEN_RECORDS <= 0:
        return records
    rec = {
        "epoch": int(epoch),
        "fidelity": float(fidelity),
        "phi_r": np.array(phi_r, dtype=float).copy(),
        "phi_b": np.array(phi_b, dtype=float).copy(),
        "amp_r": np.array(amp_r, dtype=float).copy(),
        "amp_b": np.array(amp_b, dtype=float).copy(),
    }
    filtered = [r for r in records if int(r["epoch"]) != int(epoch)]
    filtered.append(rec)
    filtered.sort(key=lambda x: float(x["fidelity"]), reverse=True)
    return filtered[: max(1, TRAIN_FID_SCREEN_RECORDS)]


def _phase_rms_distance(a, b):
    da = np.asarray(a, dtype=float)
    db = np.asarray(b, dtype=float)
    delta = np.angle(np.exp(1j * (da - db)))
    return float(np.sqrt(np.mean(delta ** 2)))


def _amp_rms_distance(a, b):
    da = np.asarray(a, dtype=float)
    db = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((da - db) ** 2)))


def _center_distance_metrics(a, b):
    phase_r = _phase_rms_distance(a["phi_r"], b["phi_r"])
    phase_b = _phase_rms_distance(a["phi_b"], b["phi_b"])
    amp_r = _amp_rms_distance(a["amp_r"], b["amp_r"])
    amp_b = _amp_rms_distance(a["amp_b"], b["amp_b"])
    phase_rms = 0.5 * (phase_r + phase_b)
    amp_rms = 0.5 * (amp_r + amp_b)
    return phase_rms, amp_rms


def _select_diverse_refined_centers(refined_centers, max_count):
    if max_count <= 0:
        return []
    ranked = sorted(refined_centers, key=lambda r: float(r["fidelity"]), reverse=True)
    if not FINAL_REFINE_DIVERSE_CENTERS or len(ranked) <= max_count:
        return ranked[:max_count]

    selected = [ranked[0]]
    remaining = ranked[1:]
    while remaining and len(selected) < max_count:
        next_idx = None
        next_score = -np.inf
        for ridx, rec in enumerate(remaining):
            min_div = np.inf
            keep = True
            for sel in selected:
                phase_rms, amp_rms = _center_distance_metrics(rec, sel)
                # Candidate is considered too close only if both metrics are small.
                if (
                    phase_rms < FINAL_REFINE_DIVERSITY_PHASE_RMS
                    and amp_rms < FINAL_REFINE_DIVERSITY_AMP_RMS
                ):
                    keep = False
                    break
                phase_norm = phase_rms / max(FINAL_REFINE_DIVERSITY_PHASE_RMS, 1.0e-12)
                amp_norm = amp_rms / max(FINAL_REFINE_DIVERSITY_AMP_RMS, 1.0e-12)
                min_div = min(min_div, phase_norm + amp_norm)
            if keep:
                # Prefer higher-fidelity and more diverse candidates jointly.
                score = float(rec["fidelity"]) + 1.0e-4 * float(min_div)
                if score > next_score:
                    next_score = score
                    next_idx = ridx
        if next_idx is None:
            break
        selected.append(remaining.pop(next_idx))

    if len(selected) < max_count:
        selected_labels = {str(r["label"]) for r in selected}
        for rec in ranked:
            if str(rec["label"]) in selected_labels:
                continue
            selected.append(rec)
            selected_labels.add(str(rec["label"]))
            if len(selected) >= max_count:
                break
    return selected[:max_count]


def _deadline_remaining(deadline_ts):
    if deadline_ts is None:
        return np.inf
    return float(deadline_ts - time.time())


def _deadline_exhausted(deadline_ts, reserve_sec=0.0):
    return _deadline_remaining(deadline_ts) <= max(0.0, float(reserve_sec))


done = False
eval_log_path = os.path.join(os.getcwd(), "eval_fidelity.csv")
if os.environ.get("CLEAR_EVAL_LOG", "1") == "1" and os.path.exists(eval_log_path):
    os.remove(eval_log_path)
best_eval_fidelity = -np.inf
best_eval_epoch = -1
best_eval_action = None
best_train_reward = -np.inf
best_train_epoch = -1
best_train_action = None
best_train_fidelity = -np.inf
best_train_fidelity_epoch = -1
best_train_fidelity_action = None
top_eval_actions = []
top_train_fidelity_actions = []
last_train_delta = None
last_reward_objective = None
reward_switch_block_logged = False
while not done:
    message, done = client_socket.recv_data()
    logger.info("Received message from RL agent server.")
    logger.info("Time stamp: %f", time.time())

    if done:
        logger.info("Training finished.")
        break

    epoch_type = message["epoch_type"]
    fidelity_data = None

    if epoch_type == "final":
        logger.info("Final Epoch")
        reward_objective = _reward_schedule_state["active"]
        logger.info(
            "Final-stage reward objective used for reporting/simulation: %s",
            reward_objective,
        )
        post_rl_start_ts = time.time()
        post_rl_deadline_ts = None
        if POST_RL_BUDGET_SEC > 0.0:
            post_rl_deadline_ts = post_rl_start_ts + POST_RL_BUDGET_SEC
            logger.info(
                "Post-RL budget active: %.1fs total (buffer %.1fs, min_stage %.1fs)",
                POST_RL_BUDGET_SEC,
                POST_RL_STOP_BUFFER_SEC,
                POST_RL_MIN_STAGE_SEC,
            )
        locs = message["locs"]
        scales = message["scales"]
        for key in locs.keys():
            logger.info("locs[%s]:", key)
            logger.info(locs[key][0])
            logger.info("scales[%s]:", key)
            logger.info(scales[key][0])
        loc_phi_r = np.array(locs["phi_r"][0], dtype=float)
        loc_phi_b = np.array(locs["phi_b"][0], dtype=float)
        scale_phi_r = np.array(scales["phi_r"][0], dtype=float)
        scale_phi_b = np.array(scales["phi_b"][0], dtype=float)
        loc_amp_r = np.array(locs.get("amp_r", [np.ones(N_SEGMENTS)])[0], dtype=float)
        loc_amp_b = np.array(locs.get("amp_b", [np.ones(N_SEGMENTS)])[0], dtype=float)
        scale_amp_r = np.array(
            scales.get("amp_r", [np.full(N_SEGMENTS, FINAL_REFINE_INIT_SIGMA_AMP)])[0],
            dtype=float,
        )
        scale_amp_b = np.array(
            scales.get("amp_b", [np.full(N_SEGMENTS, FINAL_REFINE_INIT_SIGMA_AMP)])[0],
            dtype=float,
        )

        if best_eval_action is not None:
            logger.info(
                "Using best evaluation action from epoch %d with eval fidelity %.6f",
                best_eval_epoch,
                best_eval_fidelity,
            )
            base_phi_r = np.array(best_eval_action["phi_r"], dtype=float)
            base_phi_b = np.array(best_eval_action["phi_b"], dtype=float)
            base_amp_r = np.array(best_eval_action["amp_r"], dtype=float)
            base_amp_b = np.array(best_eval_action["amp_b"], dtype=float)
        else:
            base_phi_r = loc_phi_r
            base_phi_b = loc_phi_b
            base_amp_r = loc_amp_r
            base_amp_b = loc_amp_b
        if best_train_fidelity_action is not None:
            logger.info(
                "Best screened training fidelity center: epoch=%d fidelity=%.6f",
                best_train_fidelity_epoch,
                best_train_fidelity,
            )

        # Multi-round local refinement in phase/amplitude space. This directly optimizes
        # fidelity around one or more centers using batched simulator calls.
        sigma_phi_r = np.maximum(np.asarray(scale_phi_r, dtype=float), FINAL_REFINE_MIN_SIGMA)
        sigma_phi_b = np.maximum(np.asarray(scale_phi_b, dtype=float), FINAL_REFINE_MIN_SIGMA)
        sigma_amp_r = np.maximum(
            np.asarray(scale_amp_r, dtype=float), FINAL_REFINE_MIN_SIGMA_AMP
        )
        sigma_amp_b = np.maximum(
            np.asarray(scale_amp_b, dtype=float), FINAL_REFINE_MIN_SIGMA_AMP
        )
        centers = [(base_phi_r, base_phi_b, base_amp_r, base_amp_b, "best_eval")]
        warmstart_path = _resolve_optional_npz_path(INIT_PULSES_NPZ)
        checkpoint_path = _resolve_optional_npz_path(FINAL_REFINE_CHECKPOINT_PULSES)
        same_source = (
            warmstart_path != ""
            and checkpoint_path != ""
            and warmstart_path == checkpoint_path
            and abs(INIT_PULSE_BLEND - FINAL_REFINE_CHECKPOINT_BLEND) <= 1.0e-12
        )
        if FINAL_REFINE_USE_CHECKPOINT_CENTER:
            checkpoint_center = _load_pulse_center_from_npz(
                FINAL_REFINE_CHECKPOINT_PULSES,
                FINAL_REFINE_CHECKPOINT_BLEND,
                "checkpoint",
            )
            if checkpoint_center is not None:
                centers.append((*checkpoint_center, "checkpoint_best"))
        if FINAL_REFINE_USE_WARMSTART_CENTER and INIT_PULSES_NPZ:
            if same_source:
                logger.info(
                    "Skip warmstart center: same pulse source/blend as checkpoint center (%s)",
                    warmstart_path,
                )
            else:
                warmstart_center = _load_pulse_center_from_npz(
                    INIT_PULSES_NPZ,
                    INIT_PULSE_BLEND,
                    "warmstart",
                )
                if warmstart_center is not None:
                    centers.append((*warmstart_center, "warmstart_init"))
        if FINAL_REFINE_USE_LOC_CENTER:
            centers.append((loc_phi_r, loc_phi_b, loc_amp_r, loc_amp_b, "final_loc"))
        if FINAL_REFINE_USE_TRAIN_CENTER and best_train_action is not None:
            centers.append(
                (
                    np.array(best_train_action["phi_r"], dtype=float),
                    np.array(best_train_action["phi_b"], dtype=float),
                    np.array(best_train_action["amp_r"], dtype=float),
                    np.array(best_train_action["amp_b"], dtype=float),
                    "best_train_reward",
                )
            )
        if FINAL_REFINE_USE_TRAIN_FID_CENTER and best_train_fidelity_action is not None:
            centers.append(
                (
                    np.array(best_train_fidelity_action["phi_r"], dtype=float),
                    np.array(best_train_fidelity_action["phi_b"], dtype=float),
                    np.array(best_train_fidelity_action["amp_r"], dtype=float),
                    np.array(best_train_fidelity_action["amp_b"], dtype=float),
                    "best_train_fidelity",
                )
            )
        if FINAL_REFINE_TOP_EVAL_CENTERS > 0 and top_eval_actions:
            added_top_eval = 0
            for rec in top_eval_actions:
                if best_eval_action is not None and int(rec["epoch"]) == int(best_eval_epoch):
                    continue
                centers.append(
                    (
                        np.array(rec["phi_r"], dtype=float),
                        np.array(rec["phi_b"], dtype=float),
                        np.array(rec["amp_r"], dtype=float),
                        np.array(rec["amp_b"], dtype=float),
                        f"top_eval_epoch_{int(rec['epoch'])}",
                    )
                )
                added_top_eval += 1
            logger.info(
                "Added %d historical top-eval centers for refinement",
                added_top_eval,
            )
        if (
            FINAL_REFINE_USE_TRAIN_FID_CENTER
            and TRAIN_FID_SCREEN_RECORDS > 0
            and top_train_fidelity_actions
        ):
            added_top_train_fid = 0
            for rec in top_train_fidelity_actions:
                if (
                    best_train_fidelity_action is not None
                    and int(rec["epoch"]) == int(best_train_fidelity_epoch)
                ):
                    continue
                centers.append(
                    (
                        np.array(rec["phi_r"], dtype=float),
                        np.array(rec["phi_b"], dtype=float),
                        np.array(rec["amp_r"], dtype=float),
                        np.array(rec["amp_b"], dtype=float),
                        f"top_train_fid_epoch_{int(rec['epoch'])}",
                    )
                )
                added_top_train_fid += 1
            logger.info(
                "Added %d historical top-train-fidelity centers for refinement",
                added_top_train_fid,
            )

        global_best = None
        global_best_fidelity = -np.inf
        global_best_label = "none"
        refined_centers = []
        for cidx, (c_phi_r, c_phi_b, c_amp_r, c_amp_b, label) in enumerate(centers):
            if _deadline_exhausted(
                post_rl_deadline_ts,
                POST_RL_STOP_BUFFER_SEC
                + POST_RL_MIN_STAGE_SEC
                + POST_RL_COORD_RESERVE_SEC,
            ):
                logger.warning(
                    "Post-RL budget guard: skipping remaining segment-refine centers at idx=%d (remaining %.1fs)",
                    cidx,
                    _deadline_remaining(post_rl_deadline_ts),
                )
                break
            # Use an independent RNG stream per center to avoid correlated
            # candidate sets when comparing multiple refinement centers.
            center_seed = FINAL_REFINE_SEED + cidx * FINAL_REFINE_CENTER_SEED_STRIDE
            rng_ref = np.random.default_rng(center_seed)
            logger.info("Final refinement center=%s | seed=%d", label, center_seed)
            b_phi_r, b_phi_b, b_amp_r, b_amp_b, b_fid = _refine_around_center(
                c_phi_r,
                c_phi_b,
                c_amp_r,
                c_amp_b,
                sigma_phi_r,
                sigma_phi_b,
                sigma_amp_r,
                sigma_amp_b,
                rng_ref,
                deadline_ts=post_rl_deadline_ts,
                stop_buffer_sec=POST_RL_STOP_BUFFER_SEC + POST_RL_COORD_RESERVE_SEC,
            )
            logger.info("Final refinement center=%s | best fidelity %.6f", label, b_fid)
            refined_centers.append(
                {
                    "label": label,
                    "fidelity": float(b_fid),
                    "phi_r": np.asarray(b_phi_r, dtype=float).copy(),
                    "phi_b": np.asarray(b_phi_b, dtype=float).copy(),
                    "amp_r": np.asarray(b_amp_r, dtype=float).copy(),
                    "amp_b": np.asarray(b_amp_b, dtype=float).copy(),
                }
            )
            if b_fid > global_best_fidelity:
                global_best_fidelity = b_fid
                global_best = (b_phi_r, b_phi_b, b_amp_r, b_amp_b)
                global_best_label = label

        if global_best is not None:
            base_phi_r, base_phi_b, base_amp_r, base_amp_b = global_best
            logger.info(
                "Final refinement selected center=%s with best sampled fidelity %.6f",
                global_best_label,
                global_best_fidelity,
            )
        if not refined_centers:
            refined_centers = [
                {
                    "label": global_best_label,
                    "fidelity": float(global_best_fidelity),
                    "phi_r": np.asarray(base_phi_r, dtype=float).copy(),
                    "phi_b": np.asarray(base_phi_b, dtype=float).copy(),
                    "amp_r": np.asarray(base_amp_r, dtype=float).copy(),
                    "amp_b": np.asarray(base_amp_b, dtype=float).copy(),
                }
            ]
        refined_centers.sort(key=lambda r: float(r["fidelity"]), reverse=True)

        if FINAL_REFINE_FULL_STEPS:
            n_full_candidates = min(FINAL_REFINE_FULL_CANDIDATES, len(refined_centers))
            selected_full_centers = _select_diverse_refined_centers(
                refined_centers,
                n_full_candidates,
            )
            selected_labels = ", ".join(
                f"{str(r['label'])}:{float(r['fidelity']):.6f}"
                for r in selected_full_centers
            )
            logger.info(
                "Running full-step refinement on %d selected centers (from %d refined): %s",
                len(selected_full_centers),
                len(refined_centers),
                selected_labels,
            )
            full_best_fid = -np.inf
            full_best = None
            full_best_label = "none"
            full_ranked_candidates = []
            polished_candidates = []
            n_polish_centers = (
                len(selected_full_centers)
                if FINAL_POLISH_TOP_CANDIDATES == 0
                else min(len(selected_full_centers), FINAL_POLISH_TOP_CANDIDATES)
            )
            use_scout_then_final = (
                FINAL_POLISH_ENABLE
                and n_polish_centers > 1
                and FINAL_POLISH_SCOUT_ITERS > 0
                and FINAL_POLISH_FINAL_ITERS > 0
            )
            for fidx, rec in enumerate(selected_full_centers):
                if _deadline_exhausted(
                    post_rl_deadline_ts,
                    POST_RL_STOP_BUFFER_SEC
                    + POST_RL_MIN_STAGE_SEC
                    + POST_RL_COORD_RESERVE_SEC,
                ):
                    logger.warning(
                        "Post-RL budget guard: stopping full-step center loop at rank=%d (remaining %.1fs)",
                        fidx + 1,
                        _deadline_remaining(post_rl_deadline_ts),
                    )
                    break
                label = str(rec["label"])
                full_seed = FINAL_REFINE_SEED + (97 + fidx) * FINAL_REFINE_CENTER_SEED_STRIDE
                logger.info(
                    "Starting full-step phase refinement from center=%s | seed=%d",
                    label,
                    full_seed,
                )
                (
                    phi_r_full,
                    phi_b_full,
                    amp_r_full,
                    amp_b_full,
                    full_refine_fid,
                ) = _refine_full_steps_phase(
                    rec["phi_r"],
                    rec["phi_b"],
                    rec["amp_r"],
                    rec["amp_b"],
                    sigma_phi_r,
                    sigma_phi_b,
                    sigma_amp_r,
                    sigma_amp_b,
                    np.random.default_rng(full_seed),
                    deadline_ts=post_rl_deadline_ts,
                    stop_buffer_sec=POST_RL_STOP_BUFFER_SEC + POST_RL_COORD_RESERVE_SEC,
                )
                logger.info(
                    "Full-step phase refinement center=%s | best sampled fidelity %.6f",
                    label,
                    full_refine_fid,
                )

                polish_this_center = (
                    FINAL_POLISH_ENABLE
                    and FINAL_POLISH_ITERS > 0
                    and (
                        FINAL_POLISH_TOP_CANDIDATES == 0
                        or fidx < FINAL_POLISH_TOP_CANDIDATES
                    )
                )
                if polish_this_center:
                    polish_seed = FINAL_REFINE_SEED + (
                        701 + fidx
                    ) * FINAL_REFINE_CENTER_SEED_STRIDE
                    scout_iters = (
                        FINAL_POLISH_SCOUT_ITERS
                        if use_scout_then_final
                        else FINAL_POLISH_ITERS
                    )
                    scout_patience = (
                        FINAL_POLISH_SCOUT_PATIENCE
                        if use_scout_then_final
                        else FINAL_POLISH_PATIENCE
                    )
                    if _deadline_exhausted(
                        post_rl_deadline_ts,
                        POST_RL_STOP_BUFFER_SEC + 30.0 + POST_RL_COORD_RESERVE_SEC,
                    ):
                        logger.warning(
                            "Skipping SPSA polish for center=%s due to post-RL budget (remaining %.1fs)",
                            label,
                            _deadline_remaining(post_rl_deadline_ts),
                        )
                        polish_this_center = False
                if polish_this_center:
                    logger.info(
                        "Starting SPSA polish from center=%s | seed=%d iters=%d",
                        label,
                        polish_seed,
                        scout_iters,
                    )
                    (
                        phi_r_full,
                        phi_b_full,
                        amp_r_full,
                        amp_b_full,
                        polish_fid,
                    ) = _spsa_polish_full_steps(
                        phi_r_full,
                        phi_b_full,
                        amp_r_full,
                        amp_b_full,
                        np.random.default_rng(polish_seed),
                        num_iters=scout_iters,
                        patience=scout_patience,
                        log_prefix=f"SPSA({label})",
                        allow_auto_extend=not use_scout_then_final,
                        deadline_ts=post_rl_deadline_ts,
                        stop_buffer_sec=POST_RL_STOP_BUFFER_SEC + POST_RL_COORD_RESERVE_SEC,
                    )
                    logger.info(
                        "SPSA polish center=%s | best sampled fidelity %.6f",
                        label,
                        polish_fid,
                    )
                    full_refine_fid = max(full_refine_fid, polish_fid)
                    polished_candidates.append(
                        {
                            "label": label,
                            "fidelity": float(full_refine_fid),
                            "phi_r": np.asarray(phi_r_full, dtype=float).copy(),
                            "phi_b": np.asarray(phi_b_full, dtype=float).copy(),
                            "amp_r": np.asarray(amp_r_full, dtype=float).copy(),
                            "amp_b": np.asarray(amp_b_full, dtype=float).copy(),
                        }
                    )
                else:
                    logger.info(
                        "Skipping SPSA polish for center=%s (rank=%d) due to FINAL_POLISH_TOP_CANDIDATES=%d",
                        label,
                        fidx + 1,
                        FINAL_POLISH_TOP_CANDIDATES,
                    )

                full_ranked_candidates.append(
                    {
                        "label": label,
                        "fidelity": float(full_refine_fid),
                        "phi_r": np.asarray(phi_r_full, dtype=float).copy(),
                        "phi_b": np.asarray(phi_b_full, dtype=float).copy(),
                        "amp_r": np.asarray(amp_r_full, dtype=float).copy(),
                        "amp_b": np.asarray(amp_b_full, dtype=float).copy(),
                    }
                )

                if full_refine_fid > full_best_fid:
                    full_best_fid = full_refine_fid
                    full_best = (phi_r_full, phi_b_full, amp_r_full, amp_b_full)
                    full_best_label = label

            if use_scout_then_final and polished_candidates:
                polished_candidates.sort(key=lambda x: float(x["fidelity"]), reverse=True)
                n_final_centers = min(
                    len(polished_candidates),
                    max(1, FINAL_POLISH_FINAL_TOP_CENTERS),
                )
                continuation_centers = polished_candidates[:n_final_centers]
                logger.info(
                    "Starting final SPSA continuation on %d center(s) with %d restart(s) each",
                    n_final_centers,
                    FINAL_POLISH_FINAL_RESTARTS,
                )
                stop_continuation = False
                for cidx, scout in enumerate(continuation_centers):
                    if _deadline_exhausted(
                        post_rl_deadline_ts,
                        POST_RL_STOP_BUFFER_SEC
                        + POST_RL_MIN_STAGE_SEC
                        + POST_RL_COORD_RESERVE_SEC,
                    ):
                        logger.warning(
                            "Stopping final SPSA continuation before center %d/%d due to post-RL budget (remaining %.1fs)",
                            cidx + 1,
                            n_final_centers,
                            _deadline_remaining(post_rl_deadline_ts),
                        )
                        break
                    base_label = str(scout["label"])
                    for ridx in range(max(1, FINAL_POLISH_FINAL_RESTARTS)):
                        if _deadline_exhausted(
                            post_rl_deadline_ts,
                            POST_RL_STOP_BUFFER_SEC + 30.0 + POST_RL_COORD_RESERVE_SEC,
                        ):
                            logger.warning(
                                "Stopping final SPSA continuation at center=%s restart=%d due to post-RL budget (remaining %.1fs)",
                                base_label,
                                ridx,
                                _deadline_remaining(post_rl_deadline_ts),
                            )
                            stop_continuation = True
                            break
                        seed_offset = 1701 + 101 * cidx + ridx
                        polish_seed = (
                            FINAL_REFINE_SEED
                            + seed_offset * FINAL_REFINE_CENTER_SEED_STRIDE
                        )
                        rng_polish = np.random.default_rng(polish_seed)
                        if ridx == 0:
                            phi_r_start = np.asarray(scout["phi_r"], dtype=float).copy()
                            phi_b_start = np.asarray(scout["phi_b"], dtype=float).copy()
                            amp_r_start = np.asarray(scout["amp_r"], dtype=float).copy()
                            amp_b_start = np.asarray(scout["amp_b"], dtype=float).copy()
                            restart_tag = "base"
                            phase_jitter = 0.0
                            amp_jitter = 0.0
                        else:
                            decay = FINAL_POLISH_RESTART_DECAY ** float(ridx - 1)
                            phase_jitter = FINAL_POLISH_RESTART_PHASE_JITTER * decay
                            amp_jitter = FINAL_POLISH_RESTART_AMP_JITTER * decay
                            (
                                phi_r_start,
                                phi_b_start,
                                amp_r_start,
                                amp_b_start,
                            ) = _jitter_full_steps_state(
                                scout["phi_r"],
                                scout["phi_b"],
                                scout["amp_r"],
                                scout["amp_b"],
                                phase_jitter,
                                amp_jitter,
                                rng_polish,
                            )
                            restart_tag = f"restart_{ridx}"

                        logger.info(
                            "Starting final SPSA continuation from center=%s (%s) | seed=%d iters=%d phase_jitter=%.6g amp_jitter=%.6g",
                            base_label,
                            restart_tag,
                            polish_seed,
                            FINAL_POLISH_FINAL_ITERS,
                            phase_jitter,
                            amp_jitter,
                        )
                        (
                            phi_r_polish,
                            phi_b_polish,
                            amp_r_polish,
                            amp_b_polish,
                            fid_polish,
                        ) = _spsa_polish_full_steps(
                            phi_r_start,
                            phi_b_start,
                            amp_r_start,
                            amp_b_start,
                            rng_polish,
                            num_iters=FINAL_POLISH_FINAL_ITERS,
                            patience=FINAL_POLISH_PATIENCE,
                            log_prefix=f"SPSA(final:{base_label}:{restart_tag})",
                            allow_auto_extend=True,
                            deadline_ts=post_rl_deadline_ts,
                            stop_buffer_sec=POST_RL_STOP_BUFFER_SEC
                            + POST_RL_COORD_RESERVE_SEC,
                        )
                        logger.info(
                            "Final SPSA continuation center=%s (%s) | best sampled fidelity %.6f",
                            base_label,
                            restart_tag,
                            fid_polish,
                        )
                        full_ranked_candidates.append(
                            {
                                "label": f"{base_label}:{restart_tag}",
                                "fidelity": float(fid_polish),
                                "phi_r": np.asarray(phi_r_polish, dtype=float).copy(),
                                "phi_b": np.asarray(phi_b_polish, dtype=float).copy(),
                                "amp_r": np.asarray(amp_r_polish, dtype=float).copy(),
                                "amp_b": np.asarray(amp_b_polish, dtype=float).copy(),
                            }
                        )
                        if fid_polish > full_best_fid:
                            full_best_fid = fid_polish
                            full_best = (
                                phi_r_polish,
                                phi_b_polish,
                                amp_r_polish,
                                amp_b_polish,
                            )
                            full_best_label = f"{base_label}:{restart_tag}"
                    if stop_continuation:
                        break

            if full_best is not None:
                phi_r_final, phi_b_final, amp_r_final, amp_b_final = full_best
                logger.info(
                    "Full-step refinement selected center=%s with best sampled fidelity %.6f",
                    full_best_label,
                    full_best_fid,
                )
            else:
                phi_r_final = np.repeat(base_phi_r, SEG_LEN)
                phi_b_final = np.repeat(base_phi_b, SEG_LEN)
                amp_r_final = np.repeat(base_amp_r, SEG_LEN)
                amp_b_final = np.repeat(base_amp_b, SEG_LEN)
        else:
            phi_r_final = np.repeat(base_phi_r, SEG_LEN)
            phi_b_final = np.repeat(base_phi_b, SEG_LEN)
            amp_r_final = np.repeat(base_amp_r, SEG_LEN)
            amp_b_final = np.repeat(base_amp_b, SEG_LEN)

        if FINAL_COORD_POLISH_ENABLE:
            if _deadline_exhausted(
                post_rl_deadline_ts,
                POST_RL_STOP_BUFFER_SEC + 20.0,
            ):
                logger.warning(
                    "Skipping coordinate polish due to post-RL budget (remaining %.1fs)",
                    _deadline_remaining(post_rl_deadline_ts),
                )
            else:
                coord_candidates = [
                    {
                        "label": str(full_best_label if FINAL_REFINE_FULL_STEPS else "single"),
                        "fidelity": float(full_best_fid if FINAL_REFINE_FULL_STEPS else -np.inf),
                        "phi_r": np.asarray(phi_r_final, dtype=float).copy(),
                        "phi_b": np.asarray(phi_b_final, dtype=float).copy(),
                        "amp_r": np.asarray(amp_r_final, dtype=float).copy(),
                        "amp_b": np.asarray(amp_b_final, dtype=float).copy(),
                    }
                ]
                if FINAL_COORD_POLISH_TOP_CANDIDATES > 1 and FINAL_REFINE_FULL_STEPS:
                    ranked = sorted(
                        full_ranked_candidates,
                        key=lambda x: float(x["fidelity"]),
                        reverse=True,
                    )
                    seen_labels = {str(coord_candidates[0]["label"])}
                    for rec in ranked:
                        if len(coord_candidates) >= FINAL_COORD_POLISH_TOP_CANDIDATES:
                            break
                        label = str(rec["label"])
                        if label in seen_labels:
                            continue
                        coord_candidates.append(
                            {
                                "label": label,
                                "fidelity": float(rec["fidelity"]),
                                "phi_r": np.asarray(rec["phi_r"], dtype=float).copy(),
                                "phi_b": np.asarray(rec["phi_b"], dtype=float).copy(),
                                "amp_r": np.asarray(rec["amp_r"], dtype=float).copy(),
                                "amp_b": np.asarray(rec["amp_b"], dtype=float).copy(),
                            }
                        )
                        seen_labels.add(label)
                logger.info(
                    "Starting coordinate polish portfolio: %d candidate(s) (top_candidates=%d)",
                    len(coord_candidates),
                    FINAL_COORD_POLISH_TOP_CANDIDATES,
                )
                coord_best_fid = -np.inf
                coord_best = None
                coord_best_label = "none"
                for cidx, rec in enumerate(coord_candidates):
                    if _deadline_exhausted(
                        post_rl_deadline_ts,
                        POST_RL_STOP_BUFFER_SEC + 20.0,
                    ):
                        logger.warning(
                            "Stopping coordinate polish portfolio at candidate %d/%d due to post-RL budget (remaining %.1fs)",
                            cidx + 1,
                            len(coord_candidates),
                            _deadline_remaining(post_rl_deadline_ts),
                        )
                        break
                    logger.info(
                        "Coordinate polish candidate %d/%d from center=%s | input sampled fidelity %.6f",
                        cidx + 1,
                        len(coord_candidates),
                        rec["label"],
                        float(rec["fidelity"]),
                    )
                    (
                        phi_r_coord,
                        phi_b_coord,
                        amp_r_coord,
                        amp_b_coord,
                        coord_fid,
                    ) = _coordinate_polish_full_steps(
                        rec["phi_r"],
                        rec["phi_b"],
                        rec["amp_r"],
                        rec["amp_b"],
                        deadline_ts=post_rl_deadline_ts,
                        stop_buffer_sec=POST_RL_STOP_BUFFER_SEC,
                    )
                    logger.info(
                        "Coordinate polish candidate center=%s | best sampled fidelity %.6f",
                        rec["label"],
                        coord_fid,
                    )
                    if coord_fid > coord_best_fid:
                        coord_best_fid = coord_fid
                        coord_best = (
                            np.asarray(phi_r_coord, dtype=float).copy(),
                            np.asarray(phi_b_coord, dtype=float).copy(),
                            np.asarray(amp_r_coord, dtype=float).copy(),
                            np.asarray(amp_b_coord, dtype=float).copy(),
                        )
                        coord_best_label = str(rec["label"])
                if coord_best is not None:
                    phi_r_final, phi_b_final, amp_r_final, amp_b_final = coord_best
                    logger.info(
                        "Coordinate polish selected center=%s with best sampled fidelity %.6f",
                        coord_best_label,
                        coord_best_fid,
                    )

        _, final_fidelity, rho_final, rho_target = trapped_ion_gkp_sim(
            phi_r_final,
            phi_b_final,
            amp_r=amp_r_final,
            amp_b=amp_b_final,
            n_boson=N_BOSON,
            omega=2 * np.pi * 0.002,
            t_step=T_STEP,
            alpha_cat=GKP_DELTA,
            cat_parity=GKP_LOGICAL,
            gkp_delta=GKP_DELTA,
            gkp_logical=GKP_LOGICAL,
            cat_phase=GKP_REL_PHASE,
            sample_points=FINAL_POINTS,
            target_values=FINAL_TARGET,
            sample_weights=FINAL_WEIGHTS,
            sample_area=FINAL_AREA,
            reward_scale=REWARD_SCALE,
            reward_clip=REWARD_CLIP,
            reward_norm=(
                FINAL_OVERLAP_NORM if CHAR_USE_FIXED_REWARD_NORM else FINAL_NORM
            ),
            n_shots=0,
            return_details=True,
            return_density=True,
            reward_mode="characteristic",
            characteristic_objective=reward_objective,
            gkp_squeeze_r=GKP_SQUEEZE_R,
            gkp_envelope_kappa=GKP_ENVELOPE_KAPPA,
            gkp_lattice_trunc=GKP_LATTICE_TRUNC,
        )

        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        fidelity_path = os.path.join(output_dir, "final_fidelity.txt")
        with open(fidelity_path, "w", encoding="utf-8") as f:
            f.write(f"{final_fidelity:.6f}\n")
        pulse_path = os.path.join(output_dir, "final_pulses.npz")
        np.savez(
            pulse_path,
            phi_r=phi_r_final,
            phi_b=phi_b_final,
            amp_r=amp_r_final,
            amp_b=amp_b_final,
        )
        logger.info("Final fidelity %.6f", final_fidelity)
        logger.info("Saved final fidelity to %s", fidelity_path)
        logger.info("Saved final pulses to %s", pulse_path)
        post_rl_elapsed = time.time() - post_rl_start_ts
        logger.info("Post-RL elapsed time: %.1fs", post_rl_elapsed)
        if post_rl_deadline_ts is not None:
            logger.info(
                "Post-RL budget remaining after save: %.1fs",
                _deadline_remaining(post_rl_deadline_ts),
            )

        grid = np.linspace(-PLOT_EXTENT, PLOT_EXTENT, PLOT_GRID_SIZE)
        chi_target = characteristic_grid(rho_target, grid, grid)
        chi_final = characteristic_grid(rho_final, grid, grid)

        if plt is None:
            logger.warning("matplotlib unavailable; skip characteristic plot export.")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            im0 = axes[0].imshow(
                chi_target,
                extent=[-PLOT_EXTENT, PLOT_EXTENT, -PLOT_EXTENT, PLOT_EXTENT],
                origin="lower",
                cmap="RdBu_r",
            )
            axes[0].set_title("Target GKP characteristic")
            axes[0].set_xlabel("Re(alpha)")
            axes[0].set_ylabel("Im(alpha)")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(
                chi_final,
                extent=[-PLOT_EXTENT, PLOT_EXTENT, -PLOT_EXTENT, PLOT_EXTENT],
                origin="lower",
                cmap="RdBu_r",
            )
            axes[1].set_title("Final state characteristic")
            axes[1].set_xlabel("Re(alpha)")
            axes[1].set_ylabel("Im(alpha)")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            fig.tight_layout()
            char_path = os.path.join(output_dir, "char_target_vs_final.png")
            fig.savefig(char_path, dpi=150)
            plt.close(fig)
            logger.info("Saved characteristic plot to %s", char_path)

        done = True
        logger.info("Training finished.")
        break

    action_batch = message["action_batch"]
    batch_size = message["batch_size"]
    epoch = message["epoch"]
    effective_train_epoch = _effective_train_epoch(epoch, epoch_type)
    if (
        epoch_type == "evaluation"
        and CHAR_REWARD_OBJECTIVE_STAGE2 != CHAR_REWARD_OBJECTIVE
        and CHAR_REWARD_SWITCH_EPOCH_EFFECTIVE >= 0
        and effective_train_epoch >= CHAR_REWARD_SWITCH_EPOCH_EFFECTIVE
        and (not _reward_schedule_state["switched"])
        and best_eval_fidelity < CHAR_REWARD_SWITCH_MIN_BEST_EVAL
        and (not reward_switch_block_logged)
    ):
        logger.info(
            "Reward objective switch deferred at eval epoch %d (best_eval=%.6f < min_best_eval=%.6f)",
            epoch,
            best_eval_fidelity,
            CHAR_REWARD_SWITCH_MIN_BEST_EVAL,
        )
        reward_switch_block_logged = True
    prev_reward_objective = _reward_schedule_state["active"]
    reward_objective = _update_reward_objective(epoch, epoch_type, best_eval_fidelity)
    if reward_objective != last_reward_objective:
        if (
            prev_reward_objective == CHAR_REWARD_OBJECTIVE
            and reward_objective == CHAR_REWARD_OBJECTIVE_STAGE2
        ):
            logger.info(
                "Reward objective switched at %s epoch %d -> %s (anchor_best_eval=%.6f, forced=%s, defer_eval_count=%d)",
                epoch_type,
                epoch,
                reward_objective,
                _reward_schedule_state["anchor_best_eval"],
                _reward_schedule_state["switch_forced"],
                _reward_schedule_state["defer_eval_count"],
            )
        elif (
            prev_reward_objective == CHAR_REWARD_OBJECTIVE_STAGE2
            and reward_objective == CHAR_REWARD_OBJECTIVE
            and _reward_schedule_state["reverted"]
        ):
            logger.info(
                "Reward objective reverted at %s epoch %d -> %s "
                "(stage2_best_eval=%.6f, required>=%.6f)",
                epoch_type,
                epoch,
                reward_objective,
                _reward_schedule_state["stage2_best_eval"],
                _reward_schedule_state["anchor_best_eval"] + CHAR_REWARD_STAGE2_MIN_GAIN,
            )
        else:
            logger.info(
                "Reward objective switched at %s epoch %d -> %s",
                epoch_type,
                epoch,
                reward_objective,
            )
        last_reward_objective = reward_objective

    phi_r_coeff = action_batch["phi_r"].reshape([batch_size, -1])
    phi_b_coeff = action_batch["phi_b"].reshape([batch_size, -1])
    amp_r_coeff = action_batch["amp_r"].reshape([batch_size, -1])
    amp_b_coeff = action_batch["amp_b"].reshape([batch_size, -1])

    logger.info("Start %s epoch %d", epoch_type, epoch)
    if epoch_type == "evaluation" or epoch % 20 == 0:
        _log_action_stats(
            f"Epoch {epoch} ({epoch_type})",
            phi_r_coeff,
            phi_b_coeff,
            amp_r_coeff,
            amp_b_coeff,
        )
        _log_batch_diversity(
            f"Epoch {epoch} ({epoch_type})",
            phi_r_coeff,
            phi_b_coeff,
            amp_r_coeff,
            amp_b_coeff,
        )

    rng = np.random.default_rng(epoch)
    if epoch_type == "training":
        if ACTION_NOISE_PHI > 0.0:
            phi_r_coeff = phi_r_coeff + rng.normal(0.0, ACTION_NOISE_PHI, size=phi_r_coeff.shape)
            phi_b_coeff = phi_b_coeff + rng.normal(0.0, ACTION_NOISE_PHI, size=phi_b_coeff.shape)
        if ACTION_NOISE_AMP > 0.0:
            amp_r_coeff = amp_r_coeff + rng.normal(0.0, ACTION_NOISE_AMP, size=amp_r_coeff.shape)
            amp_b_coeff = amp_b_coeff + rng.normal(0.0, ACTION_NOISE_AMP, size=amp_b_coeff.shape)

    phi_r_before = phi_r_coeff.copy()
    phi_b_before = phi_b_coeff.copy()
    amp_r_before = amp_r_coeff.copy()
    amp_b_before = amp_b_coeff.copy()
    phi_r_coeff = _project_phase(phi_r_coeff)
    phi_b_coeff = _project_phase(phi_b_coeff)
    amp_r_coeff = np.clip(amp_r_coeff, AMP_MIN, AMP_MAX)
    amp_b_coeff = np.clip(amp_b_coeff, AMP_MIN, AMP_MAX)
    if epoch_type == "evaluation" or epoch % 20 == 0:
        clip_phi_r = float(np.mean(phi_r_before != phi_r_coeff))
        clip_phi_b = float(np.mean(phi_b_before != phi_b_coeff))
        clip_amp_r = float(np.mean(amp_r_before != amp_r_coeff))
        clip_amp_b = float(np.mean(amp_b_before != amp_b_coeff))
        logger.info(
            "Clip ratio phi_r=%.3f phi_b=%.3f amp_r=%.3f amp_b=%.3f",
            clip_phi_r,
            clip_phi_b,
            clip_amp_r,
            clip_amp_b,
        )

    phi_r = np.repeat(phi_r_coeff, SEG_LEN, axis=1)
    phi_b = np.repeat(phi_b_coeff, SEG_LEN, axis=1)
    amp_r = np.repeat(amp_r_coeff, SEG_LEN, axis=1)
    amp_b = np.repeat(amp_b_coeff, SEG_LEN, axis=1)
    if epoch_type == "evaluation":
        n_shots = N_SHOTS_EVAL
        sample_points, target_values, sample_weights = EVAL_POINTS, EVAL_TARGET, EVAL_WEIGHTS
        sample_area = FINAL_BANK["char_area"]
        reward_norm = EVAL_REWARD_NORM
    else:
        n_shots = N_SHOTS_TRAIN
        train_delta = _train_delta_for_epoch(epoch)
        train_bank = TRAIN_BANKS[train_delta]
        if epoch % 100 == 0:
            logger.info(
                "Epoch %d training lattice_mix=%.3f",
                epoch,
                _lattice_mix_for_epoch(epoch),
            )
        if last_train_delta is None or train_delta != last_train_delta:
            logger.info(
                "Curriculum switched at epoch %d -> train gkp_delta=%.4f",
                epoch,
                train_delta,
            )
            last_train_delta = train_delta
        sample_points, target_values, sample_weights, reward_norm = _select_train_points(
            epoch,
            rng,
            bank=train_bank,
        )
        sample_area = train_bank["char_area"]

    if epoch_type == "evaluation":
        reward_data, fidelity_data, _, _ = trapped_ion_gkp_sim_batch(
            phi_r,
            phi_b,
            amp_r=amp_r,
            amp_b=amp_b,
            n_boson=N_BOSON,
            omega=2 * np.pi * 0.002,
            t_step=T_STEP,
            alpha_cat=GKP_DELTA,
            cat_parity=GKP_LOGICAL,
            gkp_delta=GKP_DELTA,
            gkp_logical=GKP_LOGICAL,
            cat_phase=GKP_REL_PHASE,
            sample_points=sample_points,
            target_values=target_values,
            sample_weights=sample_weights,
            sample_area=sample_area,
            reward_scale=REWARD_SCALE,
            reward_clip=REWARD_CLIP,
            reward_norm=reward_norm,
            n_shots=n_shots,
            return_details=True,
            reward_mode="characteristic",
            characteristic_objective=reward_objective,
            gkp_squeeze_r=GKP_SQUEEZE_R,
            gkp_envelope_kappa=GKP_ENVELOPE_KAPPA,
            gkp_lattice_trunc=GKP_LATTICE_TRUNC,
        )
    else:
        reward_data = trapped_ion_gkp_sim_batch(
            phi_r,
            phi_b,
            amp_r=amp_r,
            amp_b=amp_b,
            n_boson=N_BOSON,
            omega=2 * np.pi * 0.002,
            t_step=T_STEP,
            alpha_cat=GKP_DELTA,
            cat_parity=GKP_LOGICAL,
            gkp_delta=GKP_DELTA,
            gkp_logical=GKP_LOGICAL,
            cat_phase=GKP_REL_PHASE,
            sample_points=sample_points,
            target_values=target_values,
            sample_weights=sample_weights,
            sample_area=sample_area,
            reward_scale=REWARD_SCALE,
            reward_clip=REWARD_CLIP,
            reward_norm=reward_norm,
            n_shots=n_shots,
            reward_mode="characteristic",
            characteristic_objective=reward_objective,
            gkp_squeeze_r=GKP_SQUEEZE_R,
            gkp_envelope_kappa=GKP_ENVELOPE_KAPPA,
            gkp_lattice_trunc=GKP_LATTICE_TRUNC,
        )
        smooth_pen = _smoothness_penalty(phi_r, phi_b, amp_r, amp_b)
        reward_data = reward_data - SMOOTH_LAMBDA * smooth_pen

    reward_arr = np.asarray(reward_data)
    logger.info("Reward shape %s", reward_arr.shape)
    logger.info("Reward min %.6f max %.6f", float(np.min(reward_arr)), float(np.max(reward_arr)))
    R = np.mean(reward_data)
    std_R = np.std(reward_data)
    logger.info("Average reward %.3f", R)
    logger.info("STDev reward %.3f", std_R)
    if epoch_type == "training":
        best_idx_train = int(np.argmax(reward_arr))
        batch_best_reward = float(reward_arr[best_idx_train])
        if batch_best_reward > best_train_reward:
            best_train_reward = batch_best_reward
            best_train_epoch = epoch
            best_train_action = {
                "phi_r": phi_r_coeff[best_idx_train].copy(),
                "phi_b": phi_b_coeff[best_idx_train].copy(),
                "amp_r": amp_r_coeff[best_idx_train].copy(),
                "amp_b": amp_b_coeff[best_idx_train].copy(),
            }
            logger.info(
                "Updated best training action at epoch %d with reward %.6f",
                best_train_epoch,
                best_train_reward,
            )
        if (
            TRAIN_FID_SCREEN_ENABLE
            and epoch >= TRAIN_FID_SCREEN_START_EPOCH
            and (epoch % TRAIN_FID_SCREEN_INTERVAL == 0)
        ):
            screen_topk = min(TRAIN_FID_SCREEN_TOPK, int(reward_arr.shape[0]))
            if screen_topk > 0:
                top_idx_unsorted = np.argpartition(reward_arr, -screen_topk)[-screen_topk:]
                top_rewards = reward_arr[top_idx_unsorted]
                top_order = np.argsort(top_rewards)[::-1]
                top_idx = np.asarray(top_idx_unsorted[top_order], dtype=int)
                screened_fidelity = _eval_fidelity_batch(
                    phi_r_coeff[top_idx],
                    phi_b_coeff[top_idx],
                    amp_r_coeff[top_idx],
                    amp_b_coeff[top_idx],
                )
                local_best_pos = int(np.argmax(screened_fidelity))
                local_best_idx = int(top_idx[local_best_pos])
                local_best_fidelity = float(screened_fidelity[local_best_pos])
                logger.info(
                    "Train fidelity screen epoch %d | topk=%d mean=%.6f best=%.6f (batch_idx=%d reward=%.6f)",
                    epoch,
                    screen_topk,
                    float(np.mean(screened_fidelity)),
                    local_best_fidelity,
                    local_best_idx,
                    float(reward_arr[local_best_idx]),
                )
                if local_best_fidelity > best_train_fidelity:
                    best_train_fidelity = local_best_fidelity
                    best_train_fidelity_epoch = epoch
                    best_train_fidelity_action = {
                        "phi_r": phi_r_coeff[local_best_idx].copy(),
                        "phi_b": phi_b_coeff[local_best_idx].copy(),
                        "amp_r": amp_r_coeff[local_best_idx].copy(),
                        "amp_b": amp_b_coeff[local_best_idx].copy(),
                    }
                    logger.info(
                        "Updated best screened training fidelity at epoch %d: %.6f",
                        best_train_fidelity_epoch,
                        best_train_fidelity,
                    )
                top_train_fidelity_actions = _update_top_train_fidelity_actions(
                    top_train_fidelity_actions,
                    epoch=epoch,
                    fidelity=local_best_fidelity,
                    phi_r=phi_r_coeff[local_best_idx],
                    phi_b=phi_b_coeff[local_best_idx],
                    amp_r=amp_r_coeff[local_best_idx],
                    amp_b=amp_b_coeff[local_best_idx],
                )
    if fidelity_data is not None:
        fidelity_arr = np.asarray(fidelity_data, dtype=float)
        mean_fidelity = float(np.mean(fidelity_arr))
        std_fidelity = float(np.std(fidelity_arr))
        best_idx = int(np.argmax(fidelity_arr))
        batch_best_fidelity = float(fidelity_arr[best_idx])
        logger.info(
            "Eval fidelity mean %.6f | batch-best %.6f (idx=%d)",
            mean_fidelity,
            batch_best_fidelity,
            best_idx,
        )
        if batch_best_fidelity > best_eval_fidelity:
            best_eval_fidelity = batch_best_fidelity
            best_eval_epoch = epoch
            best_eval_action = {
                "phi_r": phi_r_coeff[best_idx].copy(),
                "phi_b": phi_b_coeff[best_idx].copy(),
                "amp_r": amp_r_coeff[best_idx].copy(),
                "amp_b": amp_b_coeff[best_idx].copy(),
            }
            logger.info(
                "Updated best eval action at epoch %d with fidelity %.6f",
                best_eval_epoch,
                best_eval_fidelity,
            )
        top_eval_actions = _update_top_eval_actions(
            top_eval_actions,
            epoch=epoch,
            fidelity=batch_best_fidelity,
            phi_r=phi_r_coeff[best_idx],
            phi_b=phi_b_coeff[best_idx],
            amp_r=amp_r_coeff[best_idx],
            amp_b=amp_b_coeff[best_idx],
        )
        write_header = not os.path.exists(eval_log_path)
        with open(eval_log_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write("epoch,mean_fidelity,std_fidelity\n")
            f.write(f"{epoch},{mean_fidelity:.6f},{std_fidelity:.6f}\n")

    logger.info("Sending message to RL agent server.")
    logger.info("Time stamp: %f", time.time())
    client_socket.send_data(reward_data)
