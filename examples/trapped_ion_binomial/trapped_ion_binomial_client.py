import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

if os.environ.get("DQ_FORCE_GPU", "0") == "1":
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

from quantum_control_rl_server.remote_env_tools import Client
from trapped_ion_binomial_sim_function import (
    trapped_ion_binomial_sim,
    trapped_ion_binomial_sim_batch,
    characteristic_grid,
    prepare_characteristic_distribution,
    characteristic_norm,
    binomial_target_fock_statistics,
)

logger = logging.getLogger("RL")
logger.propagate = False
logger.handlers = []
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

client_socket = Client()
(host, port) = (os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "5555")))
client_socket.connect((host, port))

FAST_SMOKE = os.environ.get("FAST_SMOKE", "0") == "1"

N_STEPS = int(os.environ.get("N_STEPS", "120"))
N_SEGMENTS = int(os.environ.get("N_SEGMENTS", "60"))
if N_STEPS <= 0 or N_SEGMENTS <= 0:
    raise ValueError("N_STEPS and N_SEGMENTS must both be positive.")
if N_STEPS % N_SEGMENTS != 0:
    raise ValueError(
        f"N_STEPS ({N_STEPS}) must be divisible by N_SEGMENTS ({N_SEGMENTS})."
    )
SEG_LEN = N_STEPS // N_SEGMENTS
T_STEP = float(os.environ.get("T_STEP", "10.0"))
if not np.isfinite(T_STEP) or T_STEP <= 0.0:
    raise ValueError(f"T_STEP must be > 0 and finite, got {T_STEP}")
OMEGA_RABI = 2 * np.pi * 0.002
SAMPLE_EXTENT = 4.0
PLOT_EXTENT = float(os.environ.get("PLOT_EXTENT", "6.0"))
N_BOSON = int(os.environ.get("N_BOSON", "30"))
_binomial_code_env = os.environ.get("BINOMIAL_CODE", "d3_z").strip()
BINOMIAL_CODE = _binomial_code_env if _binomial_code_env else "d3_z"
_binomial_phase_env = os.environ.get("BINOMIAL_REL_PHASE", "").strip()
BINOMIAL_REL_PHASE = None if _binomial_phase_env == "" else float(_binomial_phase_env)
BINOMIAL_TAIL_WARN = float(os.environ.get("BINOMIAL_TARGET_TAIL_WARN", "1.0e-3"))
BINOMIAL_TAIL_ERROR = float(os.environ.get("BINOMIAL_TARGET_TAIL_ERROR", "5.0e-3"))
ALLOW_LOW_N_BOSON = os.environ.get("ALLOW_LOW_N_BOSON", "0") == "1"

TRAIN_POINTS_STAGE1 = int(os.environ.get("TRAIN_POINTS_STAGE1", "120"))
TRAIN_POINTS_STAGE2 = int(os.environ.get("TRAIN_POINTS_STAGE2", "240"))
TRAIN_POINTS_STAGE3 = int(os.environ.get("TRAIN_POINTS_STAGE3", "960"))
TRAIN_STAGE1_EPOCHS = int(os.environ.get("TRAIN_STAGE1_EPOCHS", "120"))
TRAIN_STAGE2_EPOCHS = int(os.environ.get("TRAIN_STAGE2_EPOCHS", "240"))

CHAR_GRID_SIZE = 61
FINAL_GRID_SIZE = 61
PLOT_GRID_SIZE = 121

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

if N_BOSON <= 2:
    raise ValueError(f"N_BOSON must be > 2, got {N_BOSON}")

BINOMIAL_STATS = binomial_target_fock_statistics(
    BINOMIAL_CODE,
    N_BOSON,
    rel_phase=BINOMIAL_REL_PHASE,
)
logger.info(
    "Binomial target: code=%s rel_phase=%s mean_n=%.4f tail(n>=%d)=%.3e edge_prob=%.3e",
    BINOMIAL_CODE,
    "none" if BINOMIAL_REL_PHASE is None else f"{BINOMIAL_REL_PHASE:.3f}",
    BINOMIAL_STATS["mean_n"],
    BINOMIAL_STATS["tail_start"],
    BINOMIAL_STATS["tail_mass"],
    BINOMIAL_STATS["edge_prob"],
)
if BINOMIAL_STATS["tail_mass"] > BINOMIAL_TAIL_WARN:
    logger.warning(
        "Binomial truncation warning: tail mass %.3e above threshold %.3e",
        BINOMIAL_STATS["tail_mass"],
        BINOMIAL_TAIL_WARN,
    )
if BINOMIAL_STATS["tail_mass"] > BINOMIAL_TAIL_ERROR and not ALLOW_LOW_N_BOSON:
    raise ValueError(
        "N_BOSON appears too small for the selected binomial target "
        f"(tail mass {BINOMIAL_STATS['tail_mass']:.3e} > {BINOMIAL_TAIL_ERROR:.3e}). "
        "Increase N_BOSON or set ALLOW_LOW_N_BOSON=1 to override."
    )

SMOOTH_LAMBDA = 0.0
SMOOTH_PHI_WEIGHT = 1.0
SMOOTH_AMP_WEIGHT = 0.2
REWARD_SCALE = 1.0
REWARD_CLIP = None
CHAR_REWARD_OBJECTIVE = os.environ.get("CHAR_REWARD_OBJECTIVE", "overlap_real").lower()
CHAR_REWARD_OBJECTIVE_STAGE2 = os.environ.get(
    "CHAR_REWARD_OBJECTIVE_STAGE2", ""
).strip().lower()
CHAR_REWARD_SWITCH_EPOCH = int(os.environ.get("CHAR_REWARD_SWITCH_EPOCH", "-1"))
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
CHAR_USE_FIXED_REWARD_NORM = os.environ.get("CHAR_USE_FIXED_REWARD_NORM", "0") == "1"
CHAR_REWARD_SWITCH_MIN_BEST_EVAL = float(
    os.environ.get("CHAR_REWARD_SWITCH_MIN_BEST_EVAL", "-1.0")
)
CHAR_REWARD_STAGE2_PATIENCE_EVAL = int(
    os.environ.get("CHAR_REWARD_STAGE2_PATIENCE_EVAL", "12")
)
CHAR_REWARD_STAGE2_MIN_GAIN = float(
    os.environ.get("CHAR_REWARD_STAGE2_MIN_GAIN", "0.01")
)
CHAR_REWARD_STAGE2_ALLOW_REVERT = (
    os.environ.get("CHAR_REWARD_STAGE2_ALLOW_REVERT", "1") == "1"
)
CHAR_REWARD_AUTO_RESCALE = os.environ.get("CHAR_REWARD_AUTO_RESCALE", "1") == "1"
CHAR_REWARD_AUTO_RESCALE_TARGET_P90 = float(
    os.environ.get("CHAR_REWARD_AUTO_RESCALE_TARGET_P90", "1.0")
)
CHAR_REWARD_AUTO_RESCALE_TRIGGER_P90 = float(
    os.environ.get("CHAR_REWARD_AUTO_RESCALE_TRIGGER_P90", "3.0")
)
EVAL_INTERVAL_HINT = int(os.environ.get("EVAL_INTERVAL", "10"))
if EVAL_INTERVAL_HINT <= 0:
    raise ValueError(f"EVAL_INTERVAL must be > 0, got {EVAL_INTERVAL_HINT}")
NUM_EPOCHS_HINT = int(os.environ.get("NUM_EPOCHS", "2000"))
if NUM_EPOCHS_HINT <= 0:
    raise ValueError(f"NUM_EPOCHS must be > 0, got {NUM_EPOCHS_HINT}")
if CHAR_REWARD_STAGE2_PATIENCE_EVAL < 0:
    raise ValueError(
        f"CHAR_REWARD_STAGE2_PATIENCE_EVAL must be >= 0, got {CHAR_REWARD_STAGE2_PATIENCE_EVAL}"
    )
if not np.isfinite(CHAR_REWARD_STAGE2_MIN_GAIN) or CHAR_REWARD_STAGE2_MIN_GAIN < 0.0:
    raise ValueError(
        f"CHAR_REWARD_STAGE2_MIN_GAIN must be finite and >= 0, got {CHAR_REWARD_STAGE2_MIN_GAIN}"
    )
if (
    not np.isfinite(CHAR_REWARD_AUTO_RESCALE_TARGET_P90)
    or CHAR_REWARD_AUTO_RESCALE_TARGET_P90 <= 0.0
):
    raise ValueError(
        "CHAR_REWARD_AUTO_RESCALE_TARGET_P90 must be finite and > 0, "
        f"got {CHAR_REWARD_AUTO_RESCALE_TARGET_P90}"
    )
if (
    not np.isfinite(CHAR_REWARD_AUTO_RESCALE_TRIGGER_P90)
    or CHAR_REWARD_AUTO_RESCALE_TRIGGER_P90 <= 0.0
):
    raise ValueError(
        "CHAR_REWARD_AUTO_RESCALE_TRIGGER_P90 must be finite and > 0, "
        f"got {CHAR_REWARD_AUTO_RESCALE_TRIGGER_P90}"
    )

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
CHAR_IMPORTANCE_POWER = float(os.environ.get("CHAR_IMPORTANCE_POWER", "1.0"))
PHASE_CLIP = float(os.environ.get("PHASE_CLIP", str(np.pi)))
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
FINAL_REFINE_AMP_START_ROUND = int(os.environ.get("FINAL_REFINE_AMP_START_ROUND", "4"))
FINAL_REFINE_USE_LOC_CENTER = os.environ.get("FINAL_REFINE_USE_LOC_CENTER", "1") == "1"
FINAL_REFINE_USE_TRAIN_CENTER = os.environ.get("FINAL_REFINE_USE_TRAIN_CENTER", "1") == "1"
FINAL_REFINE_CENTER_SEED_STRIDE = int(
    os.environ.get("FINAL_REFINE_CENTER_SEED_STRIDE", "1000003")
)
FINAL_REFINE_FULL_STEPS = os.environ.get("FINAL_REFINE_FULL_STEPS", "0") == "1"
FINAL_REFINE_FULL_SAMPLES = int(os.environ.get("FINAL_REFINE_FULL_SAMPLES", "2048"))
FINAL_REFINE_FULL_ROUNDS = int(os.environ.get("FINAL_REFINE_FULL_ROUNDS", "6"))
FINAL_REFINE_FULL_TOPK = int(os.environ.get("FINAL_REFINE_FULL_TOPK", "64"))
FINAL_REFINE_FULL_SCALE = float(os.environ.get("FINAL_REFINE_FULL_SCALE", "0.6"))
FINAL_REFINE_FULL_DECAY = float(os.environ.get("FINAL_REFINE_FULL_DECAY", "0.6"))
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

ROBUST_TRAINING = os.environ.get("ROBUST_TRAINING", "0") == "1"
DEPHASE_MODEL = os.environ.get("DEPHASE_MODEL", "quasi_static").strip().lower()
DEPHASE_DETUNING_FRAC = float(os.environ.get("DEPHASE_DETUNING_FRAC", "0.05"))
DEPHASE_NOISE_SAMPLES_TRAIN = int(os.environ.get("DEPHASE_NOISE_SAMPLES_TRAIN", "6"))
DEPHASE_NOISE_SAMPLES_EVAL = int(os.environ.get("DEPHASE_NOISE_SAMPLES_EVAL", "12"))
DEPHASE_NOISE_SAMPLES_REFINE = int(os.environ.get("DEPHASE_NOISE_SAMPLES_REFINE", "16"))
DEPHASE_INCLUDE_NOMINAL = os.environ.get("DEPHASE_INCLUDE_NOMINAL", "1") == "1"
ROBUST_NOMINAL_FID_FLOOR = float(os.environ.get("ROBUST_NOMINAL_FID_FLOOR", "0.985"))
ROBUST_FLOOR_PENALTY = float(os.environ.get("ROBUST_FLOOR_PENALTY", "0.0"))
ROBUST_COMPARE_BASELINE_NPZ = os.environ.get("ROBUST_COMPARE_BASELINE_NPZ", "").strip()
DEPHASE_SWEEP_MAX_FRAC = float(os.environ.get("DEPHASE_SWEEP_MAX_FRAC", "0.08"))
DEPHASE_SWEEP_POINTS = int(os.environ.get("DEPHASE_SWEEP_POINTS", "21"))

if ROBUST_TRAINING and DEPHASE_MODEL != "quasi_static":
    raise ValueError(
        "Current robust-training stage only supports DEPHASE_MODEL=quasi_static. "
        "Stochastic segment-wise dephasing is intentionally deferred to the next phase."
    )
if not np.isfinite(DEPHASE_DETUNING_FRAC) or DEPHASE_DETUNING_FRAC < 0.0:
    raise ValueError(
        f"DEPHASE_DETUNING_FRAC must be finite and >= 0, got {DEPHASE_DETUNING_FRAC}"
    )
if DEPHASE_NOISE_SAMPLES_TRAIN <= 0:
    raise ValueError(
        f"DEPHASE_NOISE_SAMPLES_TRAIN must be > 0, got {DEPHASE_NOISE_SAMPLES_TRAIN}"
    )
if DEPHASE_NOISE_SAMPLES_EVAL <= 0:
    raise ValueError(
        f"DEPHASE_NOISE_SAMPLES_EVAL must be > 0, got {DEPHASE_NOISE_SAMPLES_EVAL}"
    )
if DEPHASE_NOISE_SAMPLES_REFINE <= 0:
    raise ValueError(
        f"DEPHASE_NOISE_SAMPLES_REFINE must be > 0, got {DEPHASE_NOISE_SAMPLES_REFINE}"
    )
if not np.isfinite(ROBUST_NOMINAL_FID_FLOOR):
    raise ValueError(
        f"ROBUST_NOMINAL_FID_FLOOR must be finite, got {ROBUST_NOMINAL_FID_FLOOR}"
    )
if not np.isfinite(ROBUST_FLOOR_PENALTY) or ROBUST_FLOOR_PENALTY < 0.0:
    raise ValueError(
        f"ROBUST_FLOOR_PENALTY must be finite and >= 0, got {ROBUST_FLOOR_PENALTY}"
    )
if not np.isfinite(DEPHASE_SWEEP_MAX_FRAC) or DEPHASE_SWEEP_MAX_FRAC <= 0.0:
    raise ValueError(
        f"DEPHASE_SWEEP_MAX_FRAC must be finite and > 0, got {DEPHASE_SWEEP_MAX_FRAC}"
    )
if DEPHASE_SWEEP_POINTS < 2:
    raise ValueError(f"DEPHASE_SWEEP_POINTS must be >= 2, got {DEPHASE_SWEEP_POINTS}")
if ROBUST_TRAINING and DEPHASE_INCLUDE_NOMINAL:
    for _name, _value in [
        ("DEPHASE_NOISE_SAMPLES_TRAIN", DEPHASE_NOISE_SAMPLES_TRAIN),
        ("DEPHASE_NOISE_SAMPLES_EVAL", DEPHASE_NOISE_SAMPLES_EVAL),
        ("DEPHASE_NOISE_SAMPLES_REFINE", DEPHASE_NOISE_SAMPLES_REFINE),
    ]:
        if _value <= 1:
            raise ValueError(
                f"{_name} must be > 1 when DEPHASE_INCLUDE_NOMINAL=1 so robust averaging has nonzero-noise samples."
            )

if FAST_SMOKE and ROBUST_TRAINING:
    DEPHASE_NOISE_SAMPLES_TRAIN = min(DEPHASE_NOISE_SAMPLES_TRAIN, 3)
    DEPHASE_NOISE_SAMPLES_EVAL = min(DEPHASE_NOISE_SAMPLES_EVAL, 5)
    DEPHASE_NOISE_SAMPLES_REFINE = min(DEPHASE_NOISE_SAMPLES_REFINE, 5)


def _build_characteristic_distribution(grid_size):
    all_points = []
    all_targets = []
    all_weights = []
    all_areas = []
    n_scales = len(CHAR_ALPHA_SCALES)
    for alpha_scale in CHAR_ALPHA_SCALES:
        points_i, target_i, weights_i, area_i = prepare_characteristic_distribution(
            n_boson=N_BOSON,
            extent=SAMPLE_EXTENT,
            grid_size=grid_size,
            binomial_code=BINOMIAL_CODE,
            mix_uniform=CHAR_UNIFORM_MIX,
            alpha_scale=alpha_scale,
            binomial_phase=BINOMIAL_REL_PHASE,
            importance_power=CHAR_IMPORTANCE_POWER,
        )
        all_points.extend(points_i)
        all_targets.append(target_i)
        all_weights.append(weights_i / float(n_scales))
        all_areas.append(area_i)
    return (
        all_points,
        np.concatenate(all_targets),
        np.concatenate(all_weights),
        float(np.mean(all_areas)),
    )


CHAR_POINTS, CHAR_TARGET, CHAR_WEIGHTS, CHAR_AREA = _build_characteristic_distribution(
    CHAR_GRID_SIZE
)
FINAL_POINTS, FINAL_TARGET, FINAL_WEIGHTS, FINAL_AREA = prepare_characteristic_distribution(
    n_boson=N_BOSON,
    extent=SAMPLE_EXTENT,
    grid_size=FINAL_GRID_SIZE,
    binomial_code=BINOMIAL_CODE,
    mix_uniform=CHAR_UNIFORM_MIX,
    alpha_scale=CHAR_ALPHA_SCALE,
    binomial_phase=BINOMIAL_REL_PHASE,
    importance_power=CHAR_IMPORTANCE_POWER,
)
CHAR_NORM = characteristic_norm(CHAR_TARGET, CHAR_AREA)
FINAL_NORM = characteristic_norm(FINAL_TARGET, FINAL_AREA)
CHAR_RADII = np.abs(np.asarray(CHAR_POINTS))

TOPK_COUNT = min(TRAIN_POINTS_STAGE1, len(CHAR_POINTS))
if CHAR_START_MODE == "topk":
    score = np.abs(CHAR_TARGET)
else:
    radii = np.maximum(CHAR_RADII, 1e-6)
    score = np.abs(CHAR_TARGET) * (radii ** CHAR_RADIAL_EXP)


def _build_stage1_topk_indices(score, count):
    # For stage-1 warmup, keep strongest informative points globally.
    return np.argsort(score)[-count:]


topk_idx = _build_stage1_topk_indices(score, TOPK_COUNT)
TOPK_POINTS = [CHAR_POINTS[i] for i in topk_idx]
TOPK_TARGET = CHAR_TARGET[topk_idx]
TOPK_WEIGHTS = np.full(TOPK_COUNT, 1.0 / TOPK_COUNT, dtype=float)
TOPK_NORM = characteristic_norm(TOPK_TARGET, CHAR_AREA)

logger.info(
    "Characteristic sampling: start_mode=%s alpha_scales=%s radial_exp=%.2f binomial_code=%s binomial_phase=%s",
    CHAR_START_MODE,
    ",".join(f"{v:.3f}" for v in CHAR_ALPHA_SCALES),
    CHAR_RADIAL_EXP,
    BINOMIAL_CODE,
    "none" if BINOMIAL_REL_PHASE is None else f"{BINOMIAL_REL_PHASE:.3f}",
)
logger.info(
    "Characteristic reward objective schedule: base=%s stage2=%s switch_train_epoch=%d | stage epochs: %d -> %d -> end",
    CHAR_REWARD_OBJECTIVE,
    CHAR_REWARD_OBJECTIVE_STAGE2,
    CHAR_REWARD_SWITCH_EPOCH,
    TRAIN_STAGE1_EPOCHS,
    TRAIN_STAGE2_EPOCHS,
)
logger.info(
    "Characteristic reward normalization: fixed_overlap_norm=%s",
    CHAR_USE_FIXED_REWARD_NORM,
)
logger.info(
    "Reward switch guard: min_best_eval=%.3f patience_eval=%d min_gain=%.4f allow_revert=%s",
    CHAR_REWARD_SWITCH_MIN_BEST_EVAL,
    CHAR_REWARD_STAGE2_PATIENCE_EVAL,
    CHAR_REWARD_STAGE2_MIN_GAIN,
    CHAR_REWARD_STAGE2_ALLOW_REVERT,
)
logger.info(
    "Reward auto-rescale: enabled=%s target_p90=%.3f trigger_p90=%.3f",
    CHAR_REWARD_AUTO_RESCALE,
    CHAR_REWARD_AUTO_RESCALE_TARGET_P90,
    CHAR_REWARD_AUTO_RESCALE_TRIGGER_P90,
)
logger.info(
    "Characteristic point sampler: mode=%s radial_bins=%d uniform_mix=%.2f importance_power=%.2f",
    CHAR_SAMPLER_MODE,
    CHAR_RADIAL_BINS,
    CHAR_UNIFORM_MIX,
    CHAR_IMPORTANCE_POWER,
)
logger.info(
    "Action clipping: phase_clip=%.3f amp_range=[%.3f, %.3f]",
    PHASE_CLIP,
    AMP_MIN,
    AMP_MAX,
)
logger.info(
    "Pulse timing: n_steps=%d n_segments=%d seg_len=%d t_step=%.6f",
    N_STEPS,
    N_SEGMENTS,
    SEG_LEN,
    T_STEP,
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
    "Full-step refinement: enabled=%s samples=%d rounds=%d topk=%d scale=%.3f decay=%.3f phase_sigma_factor=%.3f phase_min_sigma=%.4f amp_opt=%s amp_sigma_factor=%.3f amp_min_sigma=%.4f",
    FINAL_REFINE_FULL_STEPS,
    FINAL_REFINE_FULL_SAMPLES,
    FINAL_REFINE_FULL_ROUNDS,
    FINAL_REFINE_FULL_TOPK,
    FINAL_REFINE_FULL_SCALE,
    FINAL_REFINE_FULL_DECAY,
    FINAL_REFINE_FULL_SIGMA_FACTOR,
    FINAL_REFINE_FULL_MIN_SIGMA,
    FINAL_REFINE_FULL_ENABLE_AMP,
    FINAL_REFINE_FULL_SIGMA_FACTOR_AMP,
    FINAL_REFINE_FULL_MIN_SIGMA_AMP,
)
logger.info(
    "Robust dephasing: enabled=%s model=%s detuning_frac=%.4f include_nominal=%s samples(train/eval/refine)=%d/%d/%d floor=%.4f penalty=%.3f",
    ROBUST_TRAINING,
    DEPHASE_MODEL,
    DEPHASE_DETUNING_FRAC,
    DEPHASE_INCLUDE_NOMINAL,
    DEPHASE_NOISE_SAMPLES_TRAIN,
    DEPHASE_NOISE_SAMPLES_EVAL,
    DEPHASE_NOISE_SAMPLES_REFINE,
    ROBUST_NOMINAL_FID_FLOOR,
    ROBUST_FLOOR_PENALTY,
)


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
    "switch_eval_epoch": None,
    "anchor_best_eval": -np.inf,
    "stage2_best_eval": -np.inf,
}


def _update_reward_objective(epoch, epoch_type, best_eval_metric):
    active = _reward_schedule_state["active"]
    if CHAR_REWARD_OBJECTIVE_STAGE2 == CHAR_REWARD_OBJECTIVE:
        return active

    effective_train_epoch = _effective_train_epoch(epoch, epoch_type)

    # Switch only at evaluation boundaries and only once the policy is mature enough.
    can_switch = (
        not _reward_schedule_state["switched"]
        and CHAR_REWARD_SWITCH_EPOCH >= 0
        and epoch_type == "evaluation"
        and effective_train_epoch >= CHAR_REWARD_SWITCH_EPOCH
        and best_eval_metric >= CHAR_REWARD_SWITCH_MIN_BEST_EVAL
    )
    if can_switch:
        _reward_schedule_state["active"] = CHAR_REWARD_OBJECTIVE_STAGE2
        _reward_schedule_state["switched"] = True
        _reward_schedule_state["switch_eval_epoch"] = int(epoch)
        _reward_schedule_state["anchor_best_eval"] = float(best_eval_metric)
        _reward_schedule_state["stage2_best_eval"] = float(best_eval_metric)
        return _reward_schedule_state["active"]

    if (
        _reward_schedule_state["switched"]
        and (not _reward_schedule_state["reverted"])
        and epoch_type == "evaluation"
    ):
        _reward_schedule_state["stage2_best_eval"] = max(
            float(_reward_schedule_state["stage2_best_eval"]),
            float(best_eval_metric),
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


def _auto_rescale_rewards(reward_data, epoch, epoch_type):
    reward_arr = np.asarray(reward_data, dtype=float)
    if (not CHAR_REWARD_AUTO_RESCALE) or reward_arr.size == 0:
        return reward_data
    p90 = float(np.percentile(np.abs(reward_arr), 90.0))
    if (not np.isfinite(p90)) or p90 <= CHAR_REWARD_AUTO_RESCALE_TRIGGER_P90:
        return reward_data
    factor = CHAR_REWARD_AUTO_RESCALE_TARGET_P90 / p90
    scaled = reward_arr * factor
    if epoch_type == "evaluation" or int(epoch) % 20 == 0:
        logger.info(
            "Reward auto-rescale at %s epoch %d: p90=%.6f factor=%.6f",
            epoch_type,
            int(epoch),
            p90,
            factor,
        )
    return scaled.astype(np.float32)


def _expand_segments(arr):
    return np.repeat(np.asarray(arr, dtype=float), SEG_LEN, axis=1)


def _detuning_abs_max():
    return float(DEPHASE_DETUNING_FRAC * OMEGA_RABI)


def _noise_samples_for_epoch_type(epoch_type):
    if epoch_type == "evaluation":
        return max(1, DEPHASE_NOISE_SAMPLES_EVAL)
    if epoch_type == "final":
        return max(1, DEPHASE_NOISE_SAMPLES_REFINE)
    return max(1, DEPHASE_NOISE_SAMPLES_TRAIN)


def _sample_quasi_static_detuning_matrix(
    rng,
    batch_size,
    n_samples,
    include_nominal,
    shared_across_batch=True,
):
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    delta_max = _detuning_abs_max()
    if shared_across_batch:
        detuning = rng.uniform(-delta_max, delta_max, size=(1, n_samples))
        detuning = np.repeat(detuning, int(batch_size), axis=0)
    else:
        detuning = rng.uniform(-delta_max, delta_max, size=(int(batch_size), n_samples))
    if include_nominal:
        detuning[:, 0] = 0.0
    return detuning.astype(float, copy=False)


def _expand_controls_for_noise(phi_r, phi_b, amp_r, amp_b, detuning_matrix):
    batch_size, n_noise = detuning_matrix.shape
    phi_r_exp = np.repeat(np.asarray(phi_r, dtype=float), n_noise, axis=0)
    phi_b_exp = np.repeat(np.asarray(phi_b, dtype=float), n_noise, axis=0)
    amp_r_exp = np.repeat(np.asarray(amp_r, dtype=float), n_noise, axis=0)
    amp_b_exp = np.repeat(np.asarray(amp_b, dtype=float), n_noise, axis=0)
    detuning_flat = np.asarray(detuning_matrix, dtype=float).reshape(batch_size * n_noise)
    return phi_r_exp, phi_b_exp, amp_r_exp, amp_b_exp, detuning_flat


def _noise_slice_for_average(n_noise, include_nominal):
    if include_nominal and n_noise > 1:
        return slice(1, None)
    return slice(0, None)


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


def _sample_characteristic_points(rng, n_points, mode=None):
    mode = CHAR_SAMPLER_MODE if mode is None else mode
    if mode == "weighted":
        idx = rng.choice(len(CHAR_POINTS), size=n_points, replace=True, p=CHAR_WEIGHTS)
        samp_probs = CHAR_WEIGHTS[idx]
    elif mode == "uniform":
        idx = rng.choice(len(CHAR_POINTS), size=n_points, replace=True)
        samp_probs = np.full(n_points, 1.0 / len(CHAR_POINTS), dtype=float)
    elif mode == "radial_stratified":
        n_bins = max(1, CHAR_RADIAL_BINS)
        r_max = float(np.max(CHAR_RADII))
        edges = np.linspace(0.0, r_max + 1e-12, n_bins + 1)
        idx_list = []
        bin_candidates = []
        bin_mass = np.zeros(n_bins, dtype=float)
        for bi in range(n_bins):
            lo = edges[bi]
            hi = edges[bi + 1]
            if bi == n_bins - 1:
                mask = (CHAR_RADII >= lo) & (CHAR_RADII <= hi)
            else:
                mask = (CHAR_RADII >= lo) & (CHAR_RADII < hi)
            candidates = np.flatnonzero(mask)
            bin_candidates.append(candidates)
            if candidates.size > 0:
                bin_mass[bi] = float(np.sum(CHAR_WEIGHTS[candidates]))

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
        q = np.zeros(len(CHAR_POINTS), dtype=float)
        for bi in range(n_bins):
            candidates = bin_candidates[bi]
            if candidates.size == 0:
                continue
            take = min(int(quotas[bi]), n_points - len(idx_list))
            if take <= 0:
                break
            local_w = CHAR_WEIGHTS[candidates]
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
                len(CHAR_POINTS),
                size=n_fill,
                replace=True,
                p=CHAR_WEIGHTS,
            )
            idx_list.extend(fill.tolist())
            q += (n_fill / float(n_points)) * CHAR_WEIGHTS
        idx = np.asarray(idx_list, dtype=int)
        q = np.maximum(q, 1e-12)
        q = q / float(np.sum(q))
        samp_probs = q[idx]
    else:
        raise ValueError(f"Unknown CHAR_SAMPLER_MODE={mode}")
    points = [CHAR_POINTS[i] for i in idx]
    targets = CHAR_TARGET[idx]
    weights = np.asarray(samp_probs, dtype=float)
    return points, targets, weights


def _select_train_points(epoch, rng):
    if epoch < TRAIN_STAGE1_EPOCHS:
        return TOPK_POINTS, TOPK_TARGET, TOPK_WEIGHTS, TOPK_NORM
    if epoch < TRAIN_STAGE2_EPOCHS:
        points, targets, weights = _sample_characteristic_points(rng, TRAIN_POINTS_STAGE2)
        return points, targets, weights, CHAR_NORM
    points, targets, weights = _sample_characteristic_points(rng, TRAIN_POINTS_STAGE3)
    return points, targets, weights, CHAR_NORM


EVAL_RNG = np.random.default_rng(12345)
EVAL_POINTS, EVAL_TARGET, EVAL_WEIGHTS = _sample_characteristic_points(
    EVAL_RNG, TRAIN_POINTS_STAGE3, mode=CHAR_SAMPLER_MODE
)


def _eval_fidelity_batch(
    phi_r_coeff,
    phi_b_coeff,
    amp_r_coeff,
    amp_b_coeff,
    motional_detuning=0.0,
):
    phi_r_full = _expand_segments(phi_r_coeff)
    phi_b_full = _expand_segments(phi_b_coeff)
    amp_r_full = _expand_segments(amp_r_coeff)
    amp_b_full = _expand_segments(amp_b_coeff)
    return _eval_fidelity_batch_full(
        phi_r_full,
        phi_b_full,
        amp_r_full,
        amp_b_full,
        motional_detuning=motional_detuning,
    )


def _eval_fidelity_batch_full(
    phi_r_full,
    phi_b_full,
    amp_r_full,
    amp_b_full,
    motional_detuning=0.0,
):
    _, fidelity_batch, _, _ = trapped_ion_binomial_sim_batch(
        phi_r_full,
        phi_b_full,
        amp_r=amp_r_full,
        amp_b=amp_b_full,
        n_boson=N_BOSON,
        omega=OMEGA_RABI,
        t_step=T_STEP,
        binomial_code=BINOMIAL_CODE,
        binomial_phase=BINOMIAL_REL_PHASE,
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
        motional_detuning=motional_detuning,
    )
    return np.asarray(fidelity_batch, dtype=float)


def _eval_robust_reward_and_fidelity_batch_full(
    phi_r_full,
    phi_b_full,
    amp_r_full,
    amp_b_full,
    sample_points,
    target_values,
    sample_weights,
    reward_norm,
    n_shots,
    reward_objective,
    rng,
    epoch_type,
):
    batch_size = int(phi_r_full.shape[0])
    n_noise = _noise_samples_for_epoch_type(epoch_type)
    detuning_matrix = _sample_quasi_static_detuning_matrix(
        rng,
        batch_size=batch_size,
        n_samples=n_noise,
        include_nominal=DEPHASE_INCLUDE_NOMINAL,
        shared_across_batch=True,
    )
    phi_r_exp, phi_b_exp, amp_r_exp, amp_b_exp, detuning_flat = _expand_controls_for_noise(
        phi_r_full,
        phi_b_full,
        amp_r_full,
        amp_b_full,
        detuning_matrix,
    )
    reward_flat, fidelity_flat, _, _ = trapped_ion_binomial_sim_batch(
        phi_r_exp,
        phi_b_exp,
        amp_r=amp_r_exp,
        amp_b=amp_b_exp,
        n_boson=N_BOSON,
        omega=OMEGA_RABI,
        t_step=T_STEP,
        binomial_code=BINOMIAL_CODE,
        binomial_phase=BINOMIAL_REL_PHASE,
        sample_points=sample_points,
        target_values=target_values,
        sample_weights=sample_weights,
        sample_area=CHAR_AREA,
        reward_scale=REWARD_SCALE,
        reward_clip=REWARD_CLIP,
        reward_norm=reward_norm,
        n_shots=n_shots,
        return_details=True,
        reward_mode="characteristic",
        characteristic_objective=reward_objective,
        motional_detuning=detuning_flat,
    )
    reward_matrix = np.asarray(reward_flat, dtype=float).reshape(batch_size, n_noise)
    fidelity_matrix = np.asarray(fidelity_flat, dtype=float).reshape(batch_size, n_noise)

    if DEPHASE_INCLUDE_NOMINAL:
        fidelity_nominal = fidelity_matrix[:, 0]
    else:
        fidelity_nominal = _eval_fidelity_batch_full(
            phi_r_full,
            phi_b_full,
            amp_r_full,
            amp_b_full,
            motional_detuning=0.0,
        )

    avg_slice = _noise_slice_for_average(n_noise, DEPHASE_INCLUDE_NOMINAL)
    reward_robust = np.mean(reward_matrix[:, avg_slice], axis=1)
    fidelity_robust = np.mean(fidelity_matrix[:, avg_slice], axis=1)
    penalty = ROBUST_FLOOR_PENALTY * np.maximum(
        0.0,
        ROBUST_NOMINAL_FID_FLOOR - fidelity_nominal,
    )
    reward_objective_values = reward_robust - penalty
    robust_score = fidelity_robust - penalty

    return {
        "reward_objective": reward_objective_values,
        "reward_robust": reward_robust,
        "fidelity_nominal": fidelity_nominal,
        "fidelity_robust": fidelity_robust,
        "penalty": penalty,
        "score": robust_score,
        "detuning_matrix": detuning_matrix,
    }


def _eval_robust_refine_score_batch_full(phi_r_full, phi_b_full, amp_r_full, amp_b_full, rng):
    batch_size = int(phi_r_full.shape[0])
    n_noise = max(1, DEPHASE_NOISE_SAMPLES_REFINE)
    detuning_matrix = _sample_quasi_static_detuning_matrix(
        rng,
        batch_size=batch_size,
        n_samples=n_noise,
        include_nominal=DEPHASE_INCLUDE_NOMINAL,
        shared_across_batch=True,
    )
    phi_r_exp, phi_b_exp, amp_r_exp, amp_b_exp, detuning_flat = _expand_controls_for_noise(
        phi_r_full,
        phi_b_full,
        amp_r_full,
        amp_b_full,
        detuning_matrix,
    )
    _, fidelity_flat, _, _ = trapped_ion_binomial_sim_batch(
        phi_r_exp,
        phi_b_exp,
        amp_r=amp_r_exp,
        amp_b=amp_b_exp,
        n_boson=N_BOSON,
        omega=OMEGA_RABI,
        t_step=T_STEP,
        binomial_code=BINOMIAL_CODE,
        binomial_phase=BINOMIAL_REL_PHASE,
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
        motional_detuning=detuning_flat,
    )
    fidelity_matrix = np.asarray(fidelity_flat, dtype=float).reshape(batch_size, n_noise)
    if DEPHASE_INCLUDE_NOMINAL:
        fidelity_nominal = fidelity_matrix[:, 0]
    else:
        fidelity_nominal = _eval_fidelity_batch_full(
            phi_r_full,
            phi_b_full,
            amp_r_full,
            amp_b_full,
            motional_detuning=0.0,
        )
    avg_slice = _noise_slice_for_average(n_noise, DEPHASE_INCLUDE_NOMINAL)
    fidelity_robust = np.mean(fidelity_matrix[:, avg_slice], axis=1)
    penalty = ROBUST_FLOOR_PENALTY * np.maximum(
        0.0,
        ROBUST_NOMINAL_FID_FLOOR - fidelity_nominal,
    )
    score = fidelity_robust - penalty
    return score, fidelity_nominal, fidelity_robust, penalty


def _eval_refine_objective_batch_full(phi_r_full, phi_b_full, amp_r_full, amp_b_full, rng):
    if ROBUST_TRAINING:
        score, _, _, _ = _eval_robust_refine_score_batch_full(
            phi_r_full,
            phi_b_full,
            amp_r_full,
            amp_b_full,
            rng,
        )
        return score
    return _eval_fidelity_batch_full(phi_r_full, phi_b_full, amp_r_full, amp_b_full)


def _eval_refine_objective_batch(phi_r_coeff, phi_b_coeff, amp_r_coeff, amp_b_coeff, rng):
    phi_r_full = _expand_segments(phi_r_coeff)
    phi_b_full = _expand_segments(phi_b_coeff)
    amp_r_full = _expand_segments(amp_r_coeff)
    amp_b_full = _expand_segments(amp_b_coeff)
    return _eval_refine_objective_batch_full(phi_r_full, phi_b_full, amp_r_full, amp_b_full, rng)


def _refine_full_steps(
    center_phi_r,
    center_phi_b,
    center_amp_r,
    center_amp_b,
    sigma_phi_r,
    sigma_phi_b,
    sigma_amp_r,
    sigma_amp_b,
    rng,
):
    metric_name = "robust_score" if ROBUST_TRAINING else "fidelity"
    if not FINAL_REFINE_FULL_STEPS or FINAL_REFINE_FULL_SAMPLES <= 0:
        phi_r_full = np.repeat(np.asarray(center_phi_r, dtype=float), SEG_LEN)
        phi_b_full = np.repeat(np.asarray(center_phi_b, dtype=float), SEG_LEN)
        amp_r_full = np.repeat(np.asarray(center_amp_r, dtype=float), SEG_LEN)
        amp_b_full = np.repeat(np.asarray(center_amp_b, dtype=float), SEG_LEN)
        objective_val = float(
            _eval_refine_objective_batch_full(
                phi_r_full[None, :],
                phi_b_full[None, :],
                amp_r_full[None, :],
                amp_b_full[None, :],
                rng,
            )[0]
        )
        return phi_r_full, phi_b_full, amp_r_full, amp_b_full, objective_val

    cur_phi_r = np.repeat(np.asarray(center_phi_r, dtype=float), SEG_LEN)
    cur_phi_b = np.repeat(np.asarray(center_phi_b, dtype=float), SEG_LEN)
    cur_amp_r = np.repeat(np.asarray(center_amp_r, dtype=float), SEG_LEN)
    cur_amp_b = np.repeat(np.asarray(center_amp_b, dtype=float), SEG_LEN)

    sigma_phi_r_full = np.maximum(
        np.repeat(np.asarray(sigma_phi_r, dtype=float), SEG_LEN)
        * FINAL_REFINE_FULL_SIGMA_FACTOR,
        FINAL_REFINE_FULL_MIN_SIGMA,
    )
    sigma_phi_b_full = np.maximum(
        np.repeat(np.asarray(sigma_phi_b, dtype=float), SEG_LEN)
        * FINAL_REFINE_FULL_SIGMA_FACTOR,
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

    best_phi_r = cur_phi_r.copy()
    best_phi_b = cur_phi_b.copy()
    best_amp_r = cur_amp_r.copy()
    best_amp_b = cur_amp_b.copy()
    best_objective = float(
        _eval_refine_objective_batch_full(
            best_phi_r[None, :],
            best_phi_b[None, :],
            best_amp_r[None, :],
            best_amp_b[None, :],
            rng,
        )[0]
    )

    n_rounds = max(1, FINAL_REFINE_FULL_ROUNDS)
    n_samples = max(0, FINAL_REFINE_FULL_SAMPLES)
    topk = max(1, FINAL_REFINE_FULL_TOPK)

    for ridx in range(n_rounds):
        if n_samples <= 0:
            break
        scale = FINAL_REFINE_FULL_SCALE * (FINAL_REFINE_FULL_DECAY ** ridx)
        n_cand = n_samples + 1

        cand_phi_r = np.repeat(cur_phi_r[None, :], n_cand, axis=0)
        cand_phi_b = np.repeat(cur_phi_b[None, :], n_cand, axis=0)
        cand_amp_r = np.repeat(cur_amp_r[None, :], n_cand, axis=0)
        cand_amp_b = np.repeat(cur_amp_b[None, :], n_cand, axis=0)

        cand_phi_r[1:, :] = (
            cur_phi_r[None, :]
            + scale * rng.normal(size=(n_samples, N_STEPS)) * sigma_phi_r_full[None, :]
        )
        cand_phi_b[1:, :] = (
            cur_phi_b[None, :]
            + scale * rng.normal(size=(n_samples, N_STEPS)) * sigma_phi_b_full[None, :]
        )
        if FINAL_REFINE_FULL_ENABLE_AMP:
            cand_amp_r[1:, :] = (
                cur_amp_r[None, :]
                + scale
                * rng.normal(size=(n_samples, N_STEPS))
                * sigma_amp_r_full[None, :]
            )
            cand_amp_b[1:, :] = (
                cur_amp_b[None, :]
                + scale
                * rng.normal(size=(n_samples, N_STEPS))
                * sigma_amp_b_full[None, :]
            )

        cand_phi_r = np.clip(cand_phi_r, -PHASE_CLIP, PHASE_CLIP)
        cand_phi_b = np.clip(cand_phi_b, -PHASE_CLIP, PHASE_CLIP)
        cand_amp_r = np.clip(cand_amp_r, AMP_MIN, AMP_MAX)
        cand_amp_b = np.clip(cand_amp_b, AMP_MIN, AMP_MAX)

        cand_objective = _eval_refine_objective_batch_full(
            cand_phi_r,
            cand_phi_b,
            cand_amp_r,
            cand_amp_b,
            rng,
        )
        order = np.argsort(cand_objective)[::-1]
        keep = order[: min(topk, len(order))]
        round_best_idx = int(order[0])
        round_best = float(cand_objective[round_best_idx])
        logger.info(
            "Full-step refine round %d/%d | scale=%.4f best_%s=%.6f mean_topk_%s=%.6f",
            ridx + 1,
            n_rounds,
            scale,
            metric_name,
            round_best,
            metric_name,
            float(np.mean(cand_objective[keep])),
        )

        if round_best > best_objective:
            best_objective = round_best
            best_phi_r = cand_phi_r[round_best_idx].copy()
            best_phi_b = cand_phi_b[round_best_idx].copy()
            best_amp_r = cand_amp_r[round_best_idx].copy()
            best_amp_b = cand_amp_b[round_best_idx].copy()

        cur_phi_r = best_phi_r.copy()
        cur_phi_b = best_phi_b.copy()
        cur_amp_r = best_amp_r.copy()
        cur_amp_b = best_amp_b.copy()
        sigma_phi_r_full = np.maximum(np.std(cand_phi_r[keep], axis=0), FINAL_REFINE_FULL_MIN_SIGMA)
        sigma_phi_b_full = np.maximum(np.std(cand_phi_b[keep], axis=0), FINAL_REFINE_FULL_MIN_SIGMA)
        if FINAL_REFINE_FULL_ENABLE_AMP:
            sigma_amp_r_full = np.maximum(
                np.std(cand_amp_r[keep], axis=0), FINAL_REFINE_FULL_MIN_SIGMA_AMP
            )
            sigma_amp_b_full = np.maximum(
                np.std(cand_amp_b[keep], axis=0), FINAL_REFINE_FULL_MIN_SIGMA_AMP
            )

    return best_phi_r, best_phi_b, best_amp_r, best_amp_b, best_objective


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
):
    metric_name = "robust_score" if ROBUST_TRAINING else "fidelity"
    center_phi_r = np.asarray(center_phi_r, dtype=float)
    center_phi_b = np.asarray(center_phi_b, dtype=float)
    center_amp_r = np.asarray(center_amp_r, dtype=float)
    center_amp_b = np.asarray(center_amp_b, dtype=float)
    sigma_phi_r = np.asarray(sigma_phi_r, dtype=float)
    sigma_phi_b = np.asarray(sigma_phi_b, dtype=float)
    sigma_amp_r = np.asarray(sigma_amp_r, dtype=float)
    sigma_amp_b = np.asarray(sigma_amp_b, dtype=float)

    best_phi_r = center_phi_r.copy()
    best_phi_b = center_phi_b.copy()
    best_amp_r = center_amp_r.copy()
    best_amp_b = center_amp_b.copy()
    best_objective = float(
        _eval_refine_objective_batch(
            best_phi_r[None, :],
            best_phi_b[None, :],
            best_amp_r[None, :],
            best_amp_b[None, :],
            rng,
        )[0]
    )

    n_rounds = max(1, FINAL_REFINE_ROUNDS)
    n_samples = max(0, FINAL_REFINE_SAMPLES)
    topk = max(1, FINAL_REFINE_TOPK)

    for ridx in range(n_rounds):
        if n_samples <= 0:
            break
        scale = FINAL_REFINE_SCALE * (FINAL_REFINE_DECAY ** ridx)
        amp_opt_active = FINAL_REFINE_ENABLE_AMP and (ridx >= FINAL_REFINE_AMP_START_ROUND)
        n_cand = n_samples + 1

        cand_phi_r = np.repeat(best_phi_r[None, :], n_cand, axis=0)
        cand_phi_b = np.repeat(best_phi_b[None, :], n_cand, axis=0)
        cand_amp_r = np.repeat(best_amp_r[None, :], n_cand, axis=0)
        cand_amp_b = np.repeat(best_amp_b[None, :], n_cand, axis=0)

        noise_r = rng.normal(size=(n_samples, N_SEGMENTS))
        noise_b = rng.normal(size=(n_samples, N_SEGMENTS))
        cand_phi_r[1:, :] = best_phi_r[None, :] + scale * noise_r * sigma_phi_r[None, :]
        cand_phi_b[1:, :] = best_phi_b[None, :] + scale * noise_b * sigma_phi_b[None, :]
        if amp_opt_active:
            noise_amp_r = rng.normal(size=(n_samples, N_SEGMENTS))
            noise_amp_b = rng.normal(size=(n_samples, N_SEGMENTS))
            cand_amp_r[1:, :] = best_amp_r[None, :] + scale * noise_amp_r * sigma_amp_r[None, :]
            cand_amp_b[1:, :] = best_amp_b[None, :] + scale * noise_amp_b * sigma_amp_b[None, :]
        cand_phi_r = np.clip(cand_phi_r, -PHASE_CLIP, PHASE_CLIP)
        cand_phi_b = np.clip(cand_phi_b, -PHASE_CLIP, PHASE_CLIP)
        cand_amp_r = np.clip(cand_amp_r, AMP_MIN, AMP_MAX)
        cand_amp_b = np.clip(cand_amp_b, AMP_MIN, AMP_MAX)

        cand_objective = _eval_refine_objective_batch(
            cand_phi_r,
            cand_phi_b,
            cand_amp_r,
            cand_amp_b,
            rng,
        )
        order = np.argsort(cand_objective)[::-1]
        keep = order[: min(topk, len(order))]
        round_best_idx = int(order[0])
        round_best = float(cand_objective[round_best_idx])
        logger.info(
            "Final refine round %d/%d | scale=%.4f best_%s=%.6f mean_topk_%s=%.6f",
            ridx + 1,
            n_rounds,
            scale,
            metric_name,
            round_best,
            metric_name,
            float(np.mean(cand_objective[keep])),
        )

        if round_best > best_objective:
            best_objective = round_best
            best_phi_r = cand_phi_r[round_best_idx].copy()
            best_phi_b = cand_phi_b[round_best_idx].copy()
            best_amp_r = cand_amp_r[round_best_idx].copy()
            best_amp_b = cand_amp_b[round_best_idx].copy()

        sigma_phi_r = np.maximum(np.std(cand_phi_r[keep], axis=0), FINAL_REFINE_MIN_SIGMA)
        sigma_phi_b = np.maximum(np.std(cand_phi_b[keep], axis=0), FINAL_REFINE_MIN_SIGMA)
        if amp_opt_active:
            sigma_amp_r = np.maximum(
                np.std(cand_amp_r[keep], axis=0), FINAL_REFINE_MIN_SIGMA_AMP
            )
            sigma_amp_b = np.maximum(
                np.std(cand_amp_b[keep], axis=0), FINAL_REFINE_MIN_SIGMA_AMP
            )

    return best_phi_r, best_phi_b, best_amp_r, best_amp_b, best_objective


def _update_top_eval_actions(
    records,
    epoch,
    metric,
    phi_r,
    phi_b,
    amp_r,
    amp_b,
):
    if FINAL_REFINE_TOP_EVAL_CENTERS <= 0:
        return records
    rec = {
        "epoch": int(epoch),
        "metric": float(metric),
        "phi_r": np.array(phi_r, dtype=float).copy(),
        "phi_b": np.array(phi_b, dtype=float).copy(),
        "amp_r": np.array(amp_r, dtype=float).copy(),
        "amp_b": np.array(amp_b, dtype=float).copy(),
    }
    filtered = [r for r in records if int(r["epoch"]) != int(epoch)]
    filtered.append(rec)
    filtered.sort(key=lambda x: float(x["metric"]), reverse=True)
    return filtered[: max(1, FINAL_REFINE_TOP_EVAL_CENTERS)]


def _pulse_to_full_steps(values, key):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == N_STEPS:
        return arr.copy()
    if arr.size == N_SEGMENTS:
        return np.repeat(arr, SEG_LEN)
    raise ValueError(
        f"Pulse '{key}' has length {arr.size}; expected N_STEPS={N_STEPS} or N_SEGMENTS={N_SEGMENTS}."
    )


def _load_pulses_npz_as_full(npz_path):
    required = ("phi_r", "phi_b", "amp_r", "amp_b")
    with np.load(npz_path) as pulse_data:
        missing = [k for k in required if k not in pulse_data]
        if missing:
            raise ValueError(f"Missing keys {missing} in {npz_path}")
        return tuple(_pulse_to_full_steps(pulse_data[k], k) for k in required)


def _dephasing_sweep_axis():
    frac_axis = np.linspace(-DEPHASE_SWEEP_MAX_FRAC, DEPHASE_SWEEP_MAX_FRAC, DEPHASE_SWEEP_POINTS)
    detuning_axis = frac_axis * OMEGA_RABI
    return frac_axis, detuning_axis


def _evaluate_dephasing_sweep(phi_r_full, phi_b_full, amp_r_full, amp_b_full, detuning_axis):
    detuning_axis = np.asarray(detuning_axis, dtype=float).reshape(-1)
    n_det = detuning_axis.size
    phi_r_batch = np.repeat(np.asarray(phi_r_full, dtype=float)[None, :], n_det, axis=0)
    phi_b_batch = np.repeat(np.asarray(phi_b_full, dtype=float)[None, :], n_det, axis=0)
    amp_r_batch = np.repeat(np.asarray(amp_r_full, dtype=float)[None, :], n_det, axis=0)
    amp_b_batch = np.repeat(np.asarray(amp_b_full, dtype=float)[None, :], n_det, axis=0)
    return _eval_fidelity_batch_full(
        phi_r_batch,
        phi_b_batch,
        amp_r_batch,
        amp_b_batch,
        motional_detuning=detuning_axis,
    )


def _save_dephasing_sweep_outputs(
    output_dir,
    detuning_frac_axis,
    detuning_axis,
    robust_fidelity,
    baseline_fidelity=None,
):
    robust_csv_path = os.path.join(output_dir, "dephasing_sweep_robust.csv")
    with open(robust_csv_path, "w", encoding="utf-8") as f:
        f.write("detuning_frac,detuning,robust_fidelity\n")
        for frac, delta, fid in zip(detuning_frac_axis, detuning_axis, robust_fidelity):
            f.write(f"{float(frac):.8f},{float(delta):.8e},{float(fid):.8f}\n")

    fig_r, ax_r = plt.subplots(1, 1, figsize=(7, 4))
    ax_r.plot(detuning_frac_axis, robust_fidelity, "o-", label="robust pulse")
    ax_r.set_xlabel("Detuning / Omega")
    ax_r.set_ylabel("Fidelity")
    ax_r.set_title("Robust pulse dephasing sweep")
    ax_r.grid(alpha=0.25)
    ax_r.legend(loc="best")
    fig_r.tight_layout()
    robust_png_path = os.path.join(output_dir, "dephasing_sweep_robust.png")
    fig_r.savefig(robust_png_path, dpi=150)
    plt.close(fig_r)
    logger.info("Saved robust dephasing sweep to %s and %s", robust_csv_path, robust_png_path)

    if baseline_fidelity is None:
        return

    compare_csv_path = os.path.join(output_dir, "dephasing_compare.csv")
    with open(compare_csv_path, "w", encoding="utf-8") as f:
        f.write("detuning_frac,detuning,robust_fidelity,baseline_fidelity\n")
        for frac, delta, fid_r, fid_b in zip(
            detuning_frac_axis,
            detuning_axis,
            robust_fidelity,
            baseline_fidelity,
        ):
            f.write(
                f"{float(frac):.8f},{float(delta):.8e},{float(fid_r):.8f},{float(fid_b):.8f}\n"
            )

    fig_c, ax_c = plt.subplots(1, 1, figsize=(7, 4))
    ax_c.plot(detuning_frac_axis, baseline_fidelity, "o-", label="baseline pulse")
    ax_c.plot(detuning_frac_axis, robust_fidelity, "o-", label="robust pulse")
    ax_c.set_xlabel("Detuning / Omega")
    ax_c.set_ylabel("Fidelity")
    ax_c.set_title("Dephasing robustness comparison")
    ax_c.grid(alpha=0.25)
    ax_c.legend(loc="best")
    fig_c.tight_layout()
    compare_png_path = os.path.join(output_dir, "dephasing_compare.png")
    fig_c.savefig(compare_png_path, dpi=150)
    plt.close(fig_c)
    logger.info("Saved dephasing comparison to %s and %s", compare_csv_path, compare_png_path)


done = False
eval_log_path = os.path.join(os.getcwd(), "eval_fidelity.csv")
if os.environ.get("CLEAR_EVAL_LOG", "1") == "1" and os.path.exists(eval_log_path):
    os.remove(eval_log_path)
eval_robust_log_path = os.path.join(os.getcwd(), "eval_robust_metrics.csv")
if os.environ.get("CLEAR_EVAL_LOG", "1") == "1" and os.path.exists(eval_robust_log_path):
    os.remove(eval_robust_log_path)
best_eval_fidelity = -np.inf
best_eval_score = -np.inf
best_eval_epoch = -1
best_eval_action = None
best_train_reward = -np.inf
best_train_epoch = -1
best_train_action = None
top_eval_actions = []
last_reward_objective = None
reward_switch_block_logged = False
if ROBUST_TRAINING and not ROBUST_COMPARE_BASELINE_NPZ:
    logger.warning(
        "ROBUST_COMPARE_BASELINE_NPZ is not set. Robust-vs-nonrobust sweep comparison will be skipped."
    )
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
        baseline_pulses = None
        baseline_path = ""
        if ROBUST_COMPARE_BASELINE_NPZ:
            baseline_path = (
                ROBUST_COMPARE_BASELINE_NPZ
                if os.path.isabs(ROBUST_COMPARE_BASELINE_NPZ)
                else os.path.join(os.getcwd(), ROBUST_COMPARE_BASELINE_NPZ)
            )
            if not os.path.exists(baseline_path):
                raise FileNotFoundError(f"ROBUST_COMPARE_BASELINE_NPZ not found: {baseline_path}")
            baseline_pulses = _load_pulses_npz_as_full(baseline_path)

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
        scale_amp_r = np.array(
            scales.get("amp_r", [np.full(N_SEGMENTS, FINAL_REFINE_INIT_SIGMA_AMP)])[0],
            dtype=float,
        )
        scale_amp_b = np.array(
            scales.get("amp_b", [np.full(N_SEGMENTS, FINAL_REFINE_INIT_SIGMA_AMP)])[0],
            dtype=float,
        )

        if best_eval_action is not None:
            if ROBUST_TRAINING:
                logger.info(
                    "Using best evaluation action from epoch %d with robust_score %.6f (nominal_fidelity %.6f)",
                    best_eval_epoch,
                    best_eval_score,
                    best_eval_fidelity,
                )
            else:
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
            amp_r_vals = np.array(locs.get("amp_r", [np.ones(N_SEGMENTS)])[0])
            amp_b_vals = np.array(locs.get("amp_b", [np.ones(N_SEGMENTS)])[0])
            base_amp_r = np.array(amp_r_vals, dtype=float)
            base_amp_b = np.array(amp_b_vals, dtype=float)

        # Multi-round local refinement in phase space. In robust mode this
        # optimizes robust score; otherwise it optimizes nominal fidelity.
        sigma_phi_r = np.maximum(np.asarray(scale_phi_r, dtype=float), FINAL_REFINE_MIN_SIGMA)
        sigma_phi_b = np.maximum(np.asarray(scale_phi_b, dtype=float), FINAL_REFINE_MIN_SIGMA)
        sigma_amp_r = np.maximum(np.asarray(scale_amp_r, dtype=float), FINAL_REFINE_MIN_SIGMA_AMP)
        sigma_amp_b = np.maximum(np.asarray(scale_amp_b, dtype=float), FINAL_REFINE_MIN_SIGMA_AMP)
        centers = [(base_phi_r, base_phi_b, base_amp_r, base_amp_b, "best_eval")]
        if FINAL_REFINE_USE_LOC_CENTER:
            centers.append((loc_phi_r, loc_phi_b, base_amp_r, base_amp_b, "final_loc"))
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

        metric_name = "robust_score" if ROBUST_TRAINING else "fidelity"
        global_best = None
        global_best_metric = -np.inf
        global_best_label = "none"
        for cidx, (c_phi_r, c_phi_b, c_amp_r, c_amp_b, label) in enumerate(centers):
            # Use an independent RNG stream per center to avoid correlated
            # candidate sets when comparing multiple refinement centers.
            center_seed = FINAL_REFINE_SEED + cidx * FINAL_REFINE_CENTER_SEED_STRIDE
            rng_ref = np.random.default_rng(center_seed)
            logger.info("Final refinement center=%s | seed=%d", label, center_seed)
            b_phi_r, b_phi_b, b_amp_r, b_amp_b, b_metric = _refine_around_center(
                c_phi_r,
                c_phi_b,
                c_amp_r,
                c_amp_b,
                sigma_phi_r,
                sigma_phi_b,
                sigma_amp_r,
                sigma_amp_b,
                rng_ref,
            )
            logger.info(
                "Final refinement center=%s | best %s %.6f",
                label,
                metric_name,
                b_metric,
            )
            if b_metric > global_best_metric:
                global_best_metric = b_metric
                global_best = (b_phi_r, b_phi_b, b_amp_r, b_amp_b)
                global_best_label = label

        if global_best is not None:
            base_phi_r, base_phi_b, base_amp_r, base_amp_b = global_best
            logger.info(
                "Final refinement selected center=%s with best sampled %s %.6f",
                global_best_label,
                metric_name,
                global_best_metric,
            )

        if FINAL_REFINE_FULL_STEPS:
            full_seed = FINAL_REFINE_SEED + 97 * FINAL_REFINE_CENTER_SEED_STRIDE
            logger.info("Starting full-step refinement | seed=%d", full_seed)
            (
                phi_r_final,
                phi_b_final,
                amp_r_final,
                amp_b_final,
                full_best_metric,
            ) = _refine_full_steps(
                base_phi_r,
                base_phi_b,
                base_amp_r,
                base_amp_b,
                sigma_phi_r,
                sigma_phi_b,
                sigma_amp_r,
                sigma_amp_b,
                np.random.default_rng(full_seed),
            )
            logger.info(
                "Full-step refinement best sampled %s %.6f",
                metric_name,
                full_best_metric,
            )
        else:
            phi_r_final = np.repeat(base_phi_r, SEG_LEN)
            phi_b_final = np.repeat(base_phi_b, SEG_LEN)
            amp_r_final = np.repeat(base_amp_r, SEG_LEN)
            amp_b_final = np.repeat(base_amp_b, SEG_LEN)

        reward_objective_final = _reward_schedule_state["active"]

        _, final_fidelity, rho_final, rho_target = trapped_ion_binomial_sim(
            phi_r_final,
            phi_b_final,
            amp_r=amp_r_final,
            amp_b=amp_b_final,
            n_boson=N_BOSON,
            omega=OMEGA_RABI,
            t_step=T_STEP,
            binomial_code=BINOMIAL_CODE,
            binomial_phase=BINOMIAL_REL_PHASE,
            sample_points=FINAL_POINTS,
            target_values=FINAL_TARGET,
            sample_weights=FINAL_WEIGHTS,
            sample_area=FINAL_AREA,
            reward_scale=REWARD_SCALE,
            reward_clip=REWARD_CLIP,
            reward_norm=FINAL_NORM,
            n_shots=0,
            return_details=True,
            return_density=True,
            reward_mode="characteristic",
            characteristic_objective=reward_objective_final,
        )

        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        fidelity_path = os.path.join(output_dir, "final_fidelity.txt")
        with open(fidelity_path, "w", encoding="utf-8") as f:
            f.write(f"{final_fidelity:.6f}\n")
        pulse_path = os.path.join(output_dir, "final_pulses.npz")
        np.savez(
            pulse_path,
            phi_r=np.asarray(phi_r_final, dtype=float),
            phi_b=np.asarray(phi_b_final, dtype=float),
            amp_r=np.asarray(amp_r_final, dtype=float),
            amp_b=np.asarray(amp_b_final, dtype=float),
        )
        logger.info("Final fidelity %.6f", final_fidelity)
        logger.info("Saved final fidelity to %s", fidelity_path)
        logger.info("Saved final pulses to %s", pulse_path)

        final_robust_score = None
        if ROBUST_TRAINING:
            robust_seed = FINAL_REFINE_SEED + 2027 * FINAL_REFINE_CENTER_SEED_STRIDE
            score_arr, nom_arr, rob_arr, pen_arr = _eval_robust_refine_score_batch_full(
                np.asarray(phi_r_final, dtype=float)[None, :],
                np.asarray(phi_b_final, dtype=float)[None, :],
                np.asarray(amp_r_final, dtype=float)[None, :],
                np.asarray(amp_b_final, dtype=float)[None, :],
                np.random.default_rng(robust_seed),
            )
            final_robust_score = float(score_arr[0])
            final_robust_nominal = float(nom_arr[0])
            final_robust_mean = float(rob_arr[0])
            final_robust_penalty = float(pen_arr[0])
            robust_txt = os.path.join(output_dir, "final_robust_score.txt")
            with open(robust_txt, "w", encoding="utf-8") as f:
                f.write(
                    f"score={final_robust_score:.6f},f_nom={final_robust_nominal:.6f},"
                    f"f_rob={final_robust_mean:.6f},penalty={final_robust_penalty:.6f}\n"
                )
            logger.info(
                "Final robust metrics: score=%.6f f_nom=%.6f f_rob=%.6f penalty=%.6f",
                final_robust_score,
                final_robust_nominal,
                final_robust_mean,
                final_robust_penalty,
            )
            logger.info("Saved final robust metrics to %s", robust_txt)

        checkpoint_dir = os.path.join(os.getcwd(), "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_fid_txt = os.path.join(checkpoint_dir, "final_fidelity_best.txt")
        best_pulses_npz = os.path.join(checkpoint_dir, "final_pulses_best.npz")
        prev_best = -np.inf
        if os.path.exists(best_fid_txt):
            try:
                with open(best_fid_txt, "r", encoding="utf-8") as f:
                    prev_best = float(f.read().strip())
            except Exception:
                prev_best = -np.inf
        if final_fidelity > prev_best:
            with open(best_fid_txt, "w", encoding="utf-8") as f:
                f.write(f"{final_fidelity:.6f}\n")
            np.savez(
                best_pulses_npz,
                phi_r=np.asarray(phi_r_final, dtype=float),
                phi_b=np.asarray(phi_b_final, dtype=float),
                amp_r=np.asarray(amp_r_final, dtype=float),
                amp_b=np.asarray(amp_b_final, dtype=float),
            )
            logger.info(
                "Updated checkpoint best fidelity from %.6f to %.6f",
                prev_best,
                final_fidelity,
            )
        if ROBUST_TRAINING and final_robust_score is not None:
            best_robust_txt = os.path.join(checkpoint_dir, "final_robust_score_best.txt")
            best_robust_npz = os.path.join(checkpoint_dir, "final_pulses_robust_best.npz")
            prev_robust_best = -np.inf
            if os.path.exists(best_robust_txt):
                try:
                    with open(best_robust_txt, "r", encoding="utf-8") as f:
                        prev_robust_best = float(f.read().strip())
                except Exception:
                    prev_robust_best = -np.inf
            if final_robust_score > prev_robust_best:
                with open(best_robust_txt, "w", encoding="utf-8") as f:
                    f.write(f"{final_robust_score:.6f}\n")
                np.savez(
                    best_robust_npz,
                    phi_r=np.asarray(phi_r_final, dtype=float),
                    phi_b=np.asarray(phi_b_final, dtype=float),
                    amp_r=np.asarray(amp_r_final, dtype=float),
                    amp_b=np.asarray(amp_b_final, dtype=float),
                )
                logger.info(
                    "Updated checkpoint best robust score from %.6f to %.6f",
                    prev_robust_best,
                    final_robust_score,
                )

        grid = np.linspace(-PLOT_EXTENT, PLOT_EXTENT, PLOT_GRID_SIZE)
        chi_target = characteristic_grid(rho_target, grid, grid)
        chi_final = characteristic_grid(rho_final, grid, grid)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(
            chi_target,
            extent=[-PLOT_EXTENT, PLOT_EXTENT, -PLOT_EXTENT, PLOT_EXTENT],
            origin="lower",
            cmap="RdBu_r",
        )
        axes[0].set_title("Target binomial characteristic")
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

        detuning_frac_axis, detuning_axis = _dephasing_sweep_axis()
        robust_curve = _evaluate_dephasing_sweep(
            phi_r_final,
            phi_b_final,
            amp_r_final,
            amp_b_final,
            detuning_axis,
        )
        baseline_curve = None
        if baseline_pulses is not None:
            phi_r_baseline, phi_b_baseline, amp_r_baseline, amp_b_baseline = baseline_pulses
            baseline_curve = _evaluate_dephasing_sweep(
                phi_r_baseline,
                phi_b_baseline,
                amp_r_baseline,
                amp_b_baseline,
                detuning_axis,
            )
            logger.info("Loaded baseline pulses for dephasing comparison: %s", baseline_path)
        _save_dephasing_sweep_outputs(
            output_dir,
            detuning_frac_axis,
            detuning_axis,
            robust_curve,
            baseline_curve,
        )

        done = True
        logger.info("Training finished.")
        break

    action_batch = message["action_batch"]
    batch_size = message["batch_size"]
    epoch = message["epoch"]

    phi_r_coeff = action_batch["phi_r"].reshape([batch_size, -1])
    phi_b_coeff = action_batch["phi_b"].reshape([batch_size, -1])
    amp_r_coeff = action_batch["amp_r"].reshape([batch_size, -1])
    amp_b_coeff = action_batch["amp_b"].reshape([batch_size, -1])
    best_eval_metric_for_switch = best_eval_score if ROBUST_TRAINING else best_eval_fidelity
    effective_train_epoch = _effective_train_epoch(epoch, epoch_type)
    if (
        epoch_type == "evaluation"
        and CHAR_REWARD_OBJECTIVE_STAGE2 != CHAR_REWARD_OBJECTIVE
        and CHAR_REWARD_SWITCH_EPOCH >= 0
        and effective_train_epoch >= CHAR_REWARD_SWITCH_EPOCH
        and (not _reward_schedule_state["switched"])
        and best_eval_metric_for_switch < CHAR_REWARD_SWITCH_MIN_BEST_EVAL
        and (not reward_switch_block_logged)
    ):
        logger.info(
            "Reward objective switch deferred at eval epoch %d (switch_metric=%.6f < min_best_eval=%.6f)",
            epoch,
            best_eval_metric_for_switch,
            CHAR_REWARD_SWITCH_MIN_BEST_EVAL,
        )
        reward_switch_block_logged = True
    prev_reward_objective = _reward_schedule_state["active"]
    reward_objective = _update_reward_objective(epoch, epoch_type, best_eval_metric_for_switch)
    if reward_objective != last_reward_objective:
        if (
            prev_reward_objective == CHAR_REWARD_OBJECTIVE
            and reward_objective == CHAR_REWARD_OBJECTIVE_STAGE2
        ):
            logger.info(
                "Reward objective switched at %s epoch %d -> %s (anchor_best_eval=%.6f)",
                epoch_type,
                epoch,
                reward_objective,
                _reward_schedule_state["anchor_best_eval"],
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
    phi_r_coeff = np.clip(phi_r_coeff, -PHASE_CLIP, PHASE_CLIP)
    phi_b_coeff = np.clip(phi_b_coeff, -PHASE_CLIP, PHASE_CLIP)
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
        reward_norm = CHAR_NORM if CHAR_USE_FIXED_REWARD_NORM else None
    else:
        n_shots = N_SHOTS_TRAIN
        sample_points, target_values, sample_weights, reward_norm = _select_train_points(
            epoch, rng
        )
        if not CHAR_USE_FIXED_REWARD_NORM:
            reward_norm = None

    robust_reward_rob = None
    robust_fidelity_nom = None
    robust_fidelity_rob = None
    robust_penalty = None
    robust_score = None
    if ROBUST_TRAINING:
        robust_stats = _eval_robust_reward_and_fidelity_batch_full(
            phi_r,
            phi_b,
            amp_r,
            amp_b,
            sample_points=sample_points,
            target_values=target_values,
            sample_weights=sample_weights,
            reward_norm=reward_norm,
            n_shots=n_shots,
            reward_objective=reward_objective,
            rng=rng,
            epoch_type=epoch_type,
        )
        robust_reward_rob = robust_stats["reward_robust"]
        robust_fidelity_nom = robust_stats["fidelity_nominal"]
        robust_fidelity_rob = robust_stats["fidelity_robust"]
        robust_penalty = robust_stats["penalty"]
        robust_score = robust_stats["score"]
        if epoch_type == "evaluation":
            fidelity_data = robust_fidelity_nom

        if epoch_type == "evaluation" or epoch % 20 == 0:
            detuning_matrix = robust_stats["detuning_matrix"]
            avg_slice = _noise_slice_for_average(detuning_matrix.shape[1], DEPHASE_INCLUDE_NOMINAL)
            det_eval = np.asarray(detuning_matrix[:, avg_slice], dtype=float).reshape(-1)
            logger.info(
                "Robust detuning stats (%s): max_abs=%.6e mean=%.6e std=%.6e min=%.6e max=%.6e",
                epoch_type,
                float(_detuning_abs_max()),
                float(np.mean(det_eval)),
                float(np.std(det_eval)),
                float(np.min(det_eval)),
                float(np.max(det_eval)),
            )
    else:
        if epoch_type == "evaluation":
            reward_data, fidelity_data, _, _ = trapped_ion_binomial_sim_batch(
                phi_r,
                phi_b,
                amp_r=amp_r,
                amp_b=amp_b,
                n_boson=N_BOSON,
                omega=OMEGA_RABI,
                t_step=T_STEP,
                binomial_code=BINOMIAL_CODE,
                binomial_phase=BINOMIAL_REL_PHASE,
                sample_points=sample_points,
                target_values=target_values,
                sample_weights=sample_weights,
                sample_area=CHAR_AREA,
                reward_scale=REWARD_SCALE,
                reward_clip=REWARD_CLIP,
                reward_norm=reward_norm,
                n_shots=n_shots,
                return_details=True,
                reward_mode="characteristic",
                characteristic_objective=reward_objective,
            )
        else:
            reward_data = trapped_ion_binomial_sim_batch(
                phi_r,
                phi_b,
                amp_r=amp_r,
                amp_b=amp_b,
                n_boson=N_BOSON,
                omega=OMEGA_RABI,
                t_step=T_STEP,
                binomial_code=BINOMIAL_CODE,
                binomial_phase=BINOMIAL_REL_PHASE,
                sample_points=sample_points,
                target_values=target_values,
                sample_weights=sample_weights,
                sample_area=CHAR_AREA,
                reward_scale=REWARD_SCALE,
                reward_clip=REWARD_CLIP,
                reward_norm=reward_norm,
                n_shots=n_shots,
                reward_mode="characteristic",
                characteristic_objective=reward_objective,
            )
    if epoch_type != "evaluation":
        smooth_pen = _smoothness_penalty(phi_r, phi_b, amp_r, amp_b)

    if ROBUST_TRAINING:
        reward_core = np.asarray(robust_reward_rob, dtype=float)
        if epoch_type != "evaluation":
            reward_core = reward_core - SMOOTH_LAMBDA * smooth_pen
        reward_core = _auto_rescale_rewards(reward_core, epoch=epoch, epoch_type=epoch_type)
        # Keep the floor penalty in physical units; do not rescale it.
        reward_data = np.asarray(reward_core, dtype=float) - np.asarray(robust_penalty, dtype=float)
    else:
        if epoch_type != "evaluation":
            reward_data = reward_data - SMOOTH_LAMBDA * smooth_pen
        reward_data = _auto_rescale_rewards(reward_data, epoch=epoch, epoch_type=epoch_type)

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
    if fidelity_data is not None:
        fidelity_arr = np.asarray(fidelity_data, dtype=float)
        mean_fidelity = float(np.mean(fidelity_arr))
        std_fidelity = float(np.std(fidelity_arr))
        if ROBUST_TRAINING and robust_score is not None:
            robust_reward_arr = np.asarray(robust_reward_rob, dtype=float)
            robust_fid_nom_arr = np.asarray(robust_fidelity_nom, dtype=float)
            robust_fid_rob_arr = np.asarray(robust_fidelity_rob, dtype=float)
            robust_pen_arr = np.asarray(robust_penalty, dtype=float)
            robust_score_arr = np.asarray(robust_score, dtype=float)
            best_idx = int(np.argmax(robust_score_arr))
            batch_best_score = float(robust_score_arr[best_idx])
            batch_best_nominal = float(robust_fid_nom_arr[best_idx])
            batch_best_robust = float(robust_fid_rob_arr[best_idx])
            batch_best_penalty = float(robust_pen_arr[best_idx])
            logger.info(
                "Eval fidelity nominal mean %.6f std %.6f",
                mean_fidelity,
                std_fidelity,
            )
            logger.info(
                "Eval robust mean: R_rob=%.6f F_nom=%.6f F_rob=%.6f penalty=%.6f score=%.6f",
                float(np.mean(robust_reward_arr)),
                float(np.mean(robust_fid_nom_arr)),
                float(np.mean(robust_fid_rob_arr)),
                float(np.mean(robust_pen_arr)),
                float(np.mean(robust_score_arr)),
            )
            logger.info(
                "Eval robust batch-best idx=%d: score=%.6f F_nom=%.6f F_rob=%.6f penalty=%.6f",
                best_idx,
                batch_best_score,
                batch_best_nominal,
                batch_best_robust,
                batch_best_penalty,
            )
            if batch_best_score > best_eval_score:
                best_eval_score = batch_best_score
                best_eval_fidelity = batch_best_nominal
                best_eval_epoch = epoch
                best_eval_action = {
                    "phi_r": phi_r_coeff[best_idx].copy(),
                    "phi_b": phi_b_coeff[best_idx].copy(),
                    "amp_r": amp_r_coeff[best_idx].copy(),
                    "amp_b": amp_b_coeff[best_idx].copy(),
                }
                logger.info(
                    "Updated best eval action at epoch %d with robust_score %.6f (nominal_fidelity %.6f)",
                    best_eval_epoch,
                    best_eval_score,
                    best_eval_fidelity,
                )
            top_eval_actions = _update_top_eval_actions(
                top_eval_actions,
                epoch=epoch,
                metric=batch_best_score,
                phi_r=phi_r_coeff[best_idx],
                phi_b=phi_b_coeff[best_idx],
                amp_r=amp_r_coeff[best_idx],
                amp_b=amp_b_coeff[best_idx],
            )
            write_header_robust = not os.path.exists(eval_robust_log_path)
            with open(eval_robust_log_path, "a", encoding="utf-8") as f:
                if write_header_robust:
                    f.write(
                        "epoch,mean_reward_robust,mean_fidelity_nominal,mean_fidelity_robust,mean_penalty,mean_score\n"
                    )
                f.write(
                    f"{epoch},{np.mean(robust_reward_arr):.6f},{np.mean(robust_fid_nom_arr):.6f},"
                    f"{np.mean(robust_fid_rob_arr):.6f},{np.mean(robust_pen_arr):.6f},"
                    f"{np.mean(robust_score_arr):.6f}\n"
                )
        else:
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
                best_eval_score = batch_best_fidelity
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
                metric=batch_best_fidelity,
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
