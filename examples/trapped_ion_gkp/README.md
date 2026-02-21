# Trapped-Ion GKP Example

This directory is a GKP-target variant of the trapped-ion characteristic-function RL pipeline.

## Main scripts

- `trapped_ion_gkp_training_server.py`: PPO training server.
- `trapped_ion_gkp_client.py`: remote simulation client and final refinement.
- `trapped_ion_gkp_sim_function.py`: trapped-ion simulator and GKP target/characteristic utilities.
- `run_with_logs.sh`: one-command local launcher (server + client + plots).
- `parse_trapped_ion_gkp_data.py`: training/eval curve plotting.
- `plot_trapped_ion_gkp_pulses.py`: pulse sequence plotting.
- `make_characteristic_points_gif.py`: GIF of characteristic sampling points over epochs.

## Key GKP environment variables

- `GKP_DELTA` (default `0.301`)
- `GKP_LOGICAL` (default `0`; accepted: `0`, `1`, `plus`, `minus`)
- `GKP_SQUEEZE_R` (optional; default auto `-log(GKP_DELTA)`)
- `GKP_ENVELOPE_KAPPA` (optional; default auto `GKP_DELTA`)
- `GKP_LATTICE_TRUNC` (default `4`)
- `N_BOSON` (default `40` for GKP runs)
- `GKP_TARGET_TAIL_WARN` / `GKP_TARGET_TAIL_ERROR` (Fock-tail truncation diagnostics)
- `ALLOW_LOW_N_BOSON=1` (override truncation guard if you intentionally use a small basis)
- `GKP_LATTICE_MIX` (default `0.35`; inject lattice/stabilizer anchors into sampled characteristic points)
- `GKP_LATTICE_TOPK_BOOST` (default `1.5`; boost lattice anchors during stage-1 top-k warmup)
- `GKP_LATTICE_ORDER` (default `3`; reciprocal-lattice order used for anchor generation)
- `USE_PERIODIC_PHASE_PROJECTION` (default `1`; when `PHASE_CLIP≈pi`, phases are wrapped periodically instead of hard-clipped)
- `FINAL_REFINE_ENABLE_AMP` (default `0`; phase-only final refinement, set `1` to optimize amplitudes too)
- `FINAL_REFINE_FULL_STEPS` (default `0`; enable a second post-RL full-step phase refinement after segment-level refine)
- `FINAL_REFINE_FULL_SAMPLES`, `FINAL_REFINE_FULL_ROUNDS`, `FINAL_REFINE_FULL_TOPK` (controls for full-step refine)
- `FINAL_REFINE_FULL_CANDIDATES` (how many top segment-refined centers enter full-step refinement)
- `FINAL_REFINE_USE_WARMSTART_CENTER` / `FINAL_REFINE_USE_CHECKPOINT_CENTER` (inject warm-start/checkpoint pulses as explicit refinement centers)
- `FINAL_REFINE_CHECKPOINT_PULSES`, `FINAL_REFINE_CHECKPOINT_BLEND` (checkpoint center source and blend)
- `FINAL_POLISH_ENABLE` and `FINAL_POLISH_*` (optional SPSA polish directly on full-step pulses to squeeze final fidelity)
- `CHAR_REWARD_SWITCH_AUTO_CLAMP` / `CHAR_REWARD_SWITCH_AUTO_FRACTION` (auto-clamp stage-2 switch epoch when `CHAR_REWARD_SWITCH_EPOCH` is outside short fine-tune runs)
- `FINAL_COORD_POLISH_TOP_CANDIDATES` (default `1`; run coordinate polish on top-K post-refine candidates and keep the best)
- `FINAL_COORD_POLISH_EMPTY_SWEEP_PATIENCE` (default `2`; keep shrinking coordinate-polish step size for a few empty sweeps before stopping)
- `FINAL_COORD_POLISH_RESERVE_SEC` (reserve this many post-RL seconds for coordinate polish; avoids spending all budget in earlier SPSA stages)
- `TRAIN_FID_SCREEN_ENABLE`, `TRAIN_FID_SCREEN_*` (periodically evaluate true fidelity on top-reward training candidates and keep high-fidelity centers for final refinement)
- `FINAL_REFINE_USE_TRAIN_FID_CENTER` (inject best screened training-fidelity center into final multi-center refinement)

By default the GKP target follows the finite-energy mapping used in the papers:
- envelope parameter `delta`
- peak squeezing `r = -log(delta)`
- envelope operator scale `kappa = delta`

## Recommended GKP run profile

For stable trapped-ion GKP optimization (phase-only controls aligned with the paper workflow), start from:

- `LEARN_AMP_R=0`, `LEARN_AMP_B=0`
- `NUM_EPOCHS=2000`
- `TRAIN_STAGE1_EPOCHS=0`, `TRAIN_STAGE2_EPOCHS=180`
- `CHAR_ALPHA_SCALES=1.0` (single-scale first)
- keep curriculum off initially (`unset GKP_DELTA_CURRICULUM`, `unset GKP_DELTA_CURRICULUM_EPOCHS`)

## Quick run

Run inside an activated environment on a compute node:

```bash
cd examples/trapped_ion_gkp
bash run_with_logs.sh
```

If your server/client dependencies are split across different environments,
launch with separate interpreters:

```bash
SERVER_PYTHON=/path/to/venv_tf/bin/python \
CLIENT_PYTHON=/path/to/venv_dq/bin/python \
POST_PYTHON=/path/to/venv_dq/bin/python \
bash run_with_logs.sh
```
