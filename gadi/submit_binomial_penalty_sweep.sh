#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-mu61}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/${PROJECT}/${USER}/quantum_control_rl_server}"
BINOMIAL_DIR="${PROJECT_DIR}/examples/trapped_ion_binomial"
PENALTIES_CSV="${PENALTIES_CSV:-0,0.5,1,2,4}"
SWEEP_ID="${SWEEP_ID:-penalty_sweep_$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${SWEEP_ROOT:-${BINOMIAL_DIR}/penalty_sweep/${SWEEP_ID}}"
BASELINE_NPZ="${BASELINE_NPZ:-${BINOMIAL_DIR}/checkpoint/final_pulses_best.npz}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "PROJECT_DIR not found: ${PROJECT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${BASELINE_NPZ}" ]]; then
  echo "BASELINE_NPZ not found: ${BASELINE_NPZ}" >&2
  exit 1
fi

mkdir -p "${SWEEP_ROOT}"

IFS=',' read -r -a PENALTIES <<< "${PENALTIES_CSV}"
if [[ "${#PENALTIES[@]}" -eq 0 ]]; then
  echo "No penalties parsed from PENALTIES_CSV=${PENALTIES_CSV}" >&2
  exit 1
fi

echo "Submitting binomial robust penalty sweep"
echo "  Project dir: ${PROJECT_DIR}"
echo "  Sweep root : ${SWEEP_ROOT}"
echo "  Penalties  : ${PENALTIES_CSV}"
echo "  Baseline   : ${BASELINE_NPZ}"

prev_job=""
job_ids=()

for raw_p in "${PENALTIES[@]}"; do
  p="$(echo "${raw_p}" | xargs)"
  tag="p_${p//./p}"
  deps=()
  if [[ -n "${prev_job}" ]]; then
    deps=(-W "depend=afterok:${prev_job}")
  fi
  jid="$(
    qsub "${deps[@]}" \
      -v "PROJECT=${PROJECT},PROJECT_DIR=${PROJECT_DIR},SWEEP_ROOT=${SWEEP_ROOT},SWEEP_TAG=${tag},ROBUST_TRAINING=1,DEPHASE_MODEL=quasi_static,DEPHASE_DETUNING_FRAC=0.05,DEPHASE_NOISE_SAMPLES_TRAIN=6,DEPHASE_NOISE_SAMPLES_EVAL=12,DEPHASE_NOISE_SAMPLES_REFINE=16,DEPHASE_INCLUDE_NOMINAL=1,ROBUST_NOMINAL_FID_FLOOR=0.985,ROBUST_FLOOR_PENALTY=${p},ROBUST_COMPARE_BASELINE_NPZ=${BASELINE_NPZ}" \
      "${PROJECT_DIR}/gadi/run_job_binomial_penalty_sweep.pbs"
  )"
  echo "  penalty=${p} tag=${tag} -> ${jid}"
  prev_job="${jid}"
  job_ids+=("${jid}")
done

jobs_txt="${SWEEP_ROOT}/submitted_jobs.txt"
{
  echo "SWEEP_ROOT=${SWEEP_ROOT}"
  echo "PENALTIES=${PENALTIES_CSV}"
  echo "BASELINE_NPZ=${BASELINE_NPZ}"
  for jid in "${job_ids[@]}"; do
    echo "${jid}"
  done
} > "${jobs_txt}"

cat <<EOF

Submitted ${#job_ids[@]} jobs.
Job list saved to: ${jobs_txt}

When all jobs finish, aggregate with:
  source /scratch/${PROJECT}/${USER}/qcrl_envs/venv_dq/bin/activate
  python ${PROJECT_DIR}/gadi/aggregate_binomial_penalty_sweep.py --sweep-root ${SWEEP_ROOT}

Monitor queue:
  qstat -u ${USER}
EOF
