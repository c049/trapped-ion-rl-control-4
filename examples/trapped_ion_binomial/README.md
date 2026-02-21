# Trapped-Ion Binomial Example

This directory is the binomial-target variant of the trapped-ion characteristic-function RL pipeline.

## Main scripts

- `trapped_ion_binomial_training_server.py`: PPO training server.
- `trapped_ion_binomial_client.py`: remote simulation client and final refinement.
- `trapped_ion_binomial_sim_function.py`: trapped-ion simulator and binomial target/characteristic utilities.
- `run_with_logs.sh`: one-command local launcher (server + client + plots).
- `parse_trapped_ion_binomial_data.py`: training/eval curve plotting.
- `plot_trapped_ion_binomial_pulses.py`: pulse sequence plotting.
- `make_characteristic_points_gif.py`: GIF of characteristic sampling points over epochs.

## Binomial target options

- `BINOMIAL_CODE` (default `d3_z`)
  - `d3_z`: `(sqrt(3)|3> + |9>) / 2` (PRX 2022 appendix example)
  - `s2_plus`: `(|0> + sqrt(3)|6>) / 2`
  - `s1_plus`: `(|0> + |4>) / sqrt(2)`
  - backward-compatible aliases include `d3_minus`, `d3_plus`, `s2_z`
- `BINOMIAL_REL_PHASE` (optional relative phase on the second component)
- `N_BOSON` (default `30`)
- `CHAR_IMPORTANCE_POWER` (default `1.0`; sampling density proportional to `|chi_target|^power`)
- `CHAR_REWARD_OBJECTIVE_STAGE2` and `CHAR_REWARD_SWITCH_EPOCH` (optional objective schedule)
- `CHAR_REWARD_SWITCH_MIN_BEST_EVAL`, `CHAR_REWARD_STAGE2_PATIENCE_EVAL`,
  `CHAR_REWARD_STAGE2_MIN_GAIN`, `CHAR_REWARD_STAGE2_ALLOW_REVERT`
  (guard/rollback controls for objective switching)
- `CHAR_REWARD_AUTO_RESCALE=1` with `CHAR_REWARD_AUTO_RESCALE_TARGET_P90`
  (stabilize reward scale when overlap-style rewards become too large)
- `BINOMIAL_TARGET_TAIL_WARN` / `BINOMIAL_TARGET_TAIL_ERROR` (Fock-tail diagnostics)
- `ALLOW_LOW_N_BOSON=1` (override truncation guard intentionally)
- `FINAL_REFINE_ENABLE_AMP=1` (segment-level amplitude search in final local refinement)
- `FINAL_REFINE_FULL_STEPS=1` (optional step-level final refinement)

## Dephasing-robust options (quasi-static)

- `ROBUST_TRAINING=1`: enable dephasing-robust training objective.
- `DEPHASE_MODEL` (default `quasi_static`): robust-training stage selector.
  In this stage only `quasi_static` is supported.
- `DEPHASE_DETUNING_FRAC` (default `0.05`): quasi-static detuning range as a fraction of Rabi rate (`delta in [-frac*Omega, +frac*Omega]`).
- `DEPHASE_NOISE_SAMPLES_TRAIN` / `DEPHASE_NOISE_SAMPLES_EVAL` / `DEPHASE_NOISE_SAMPLES_REFINE`:
  number of sampled dephasing trajectories for training, evaluation, and final refinement.
- `DEPHASE_INCLUDE_NOMINAL=1`: include a nominal sample (`delta=0`) in each sampled set.
- `ROBUST_NOMINAL_FID_FLOOR` (default `0.985`): nominal fidelity floor.
- `ROBUST_FLOOR_PENALTY` (default `0.0`): penalty multiplier for violating the floor.
  Keep at `0` for strict "optimize expected robust performance first" behavior.
- `DEPHASE_SWEEP_MAX_FRAC` / `DEPHASE_SWEEP_POINTS`:
  sweep range and resolution used for final fidelity-vs-dephasing curves.
- `ROBUST_COMPARE_BASELINE_NPZ`:
  optional baseline pulse file for robust-vs-nonrobust sweep comparison.

## Outputs

- `outputs/final_fidelity.txt`: final evaluated fidelity.
- `outputs/final_pulses.npz`: final pulse waveforms (`phi_r`, `phi_b`, `amp_r`, `amp_b`).
- `outputs/dephasing_sweep_robust.csv` + `outputs/dephasing_sweep_robust.png`:
  final pulse fidelity-vs-detuning curve.
- `outputs/dephasing_compare.csv` + `outputs/dephasing_compare.png`:
  robust-vs-baseline comparison (generated when `ROBUST_COMPARE_BASELINE_NPZ` is set).
- `eval_robust_metrics.csv`: robust evaluation summary (`R_rob`, `F_nom`, `F_rob`, `penalty`, `score`) per eval epoch.
- `outputs/final_robust_score.txt`: final robust score summary for the final pulse.
- `checkpoint/final_fidelity_best.txt` + `checkpoint/final_pulses_best.npz`: best-so-far run cache.
- `checkpoint/final_robust_score_best.txt` + `checkpoint/final_pulses_robust_best.npz`:
  best-so-far robust-score checkpoint in robust mode.

## Quick run

Run inside an activated environment on a compute node:

```bash
cd examples/trapped_ion_binomial
bash run_with_logs.sh
```

If your server/client dependencies are split across environments:

```bash
SERVER_PYTHON=/path/to/venv_tf/bin/python \
CLIENT_PYTHON=/path/to/venv_dq/bin/python \
POST_PYTHON=/path/to/venv_dq/bin/python \
bash run_with_logs.sh
```

Warm-start fine-tune (optional):

```bash
INIT_PULSES_NPZ=outputs/final_pulses.npz \
INIT_PULSE_BLEND=1.0 \
N_SEGMENTS=120 \
LEARN_AMP_R=1 LEARN_AMP_B=1 \
bash run_with_logs.sh
```

## Robust-vs-baseline workflow

1. Run non-robust baseline and keep its pulses:

```bash
ROBUST_TRAINING=0 \
bash run_with_logs.sh

cp outputs/final_pulses.npz checkpoint/nonrobust_baseline_pulses.npz
```

2. Run robust training and compare against the baseline pulses:

```bash
ROBUST_TRAINING=1 \
DEPHASE_MODEL=quasi_static \
DEPHASE_DETUNING_FRAC=0.05 \
DEPHASE_NOISE_SAMPLES_TRAIN=6 \
DEPHASE_NOISE_SAMPLES_EVAL=12 \
DEPHASE_NOISE_SAMPLES_REFINE=16 \
ROBUST_NOMINAL_FID_FLOOR=0.985 \
ROBUST_FLOOR_PENALTY=0.0 \
ROBUST_COMPARE_BASELINE_NPZ=checkpoint/nonrobust_baseline_pulses.npz \
bash run_with_logs.sh
```
