#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/project/TurboDiffusion/turbot2va"
JAVIS_DIR="${JAVIS_DIR:-/root/autodl-tmp/project/JavisDiT}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
TORCHRUN="${TORCHRUN:-/root/miniconda3/bin/torchrun}"
DECODE_ROOT="${DECODE_ROOT:?Set DECODE_ROOT to the decoded figure5 output directory}"
RESULTS_ROOT="${RESULTS_ROOT:-${DECODE_ROOT}_metrics}"
PROMPTS_FILE="${PROMPTS_FILE:-/root/autodl-tmp/project/00_mydata/prompts.txt}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
ASSIGNMENTS="${ASSIGNMENTS:-topk_0.5:0:29901 topk_0.4:1:29902 topk_0.3:2:29903 topk_0.2:3:29904}"
MAX_AUDIO_LEN_S="${MAX_AUDIO_LEN_S:-8.0}"
METRICS="${METRICS:-imagebind-score cxxp-score av-score desync}"
EXCLUDE="${EXCLUDE:-clip_score clap_score}"
FORCE_EVAL="${FORCE_EVAL:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-1}"

mkdir -p "${RESULTS_ROOT}"
MANIFEST="${RESULTS_ROOT}/manifest_00_mydata.csv"
"${PYTHON}" "${ROOT_DIR}/scripts/make_javis_manifest.py" \
  --prompts_file "${PROMPTS_FILE}" \
  --output_csv "${MANIFEST}" \
  --num_prompts "${NUM_PROMPTS}"

cd "${JAVIS_DIR}"
pids=()
configs=()
for assignment in ${ASSIGNMENTS}; do
  IFS=: read -r config_name gpu master_port <<< "${assignment}"
  infer_dir="${DECODE_ROOT}/${config_name}"
  output_file="${RESULTS_ROOT}/${config_name}.json"
  log_file="${RESULTS_ROOT}/${config_name}.log"
  if [ ! -d "${infer_dir}" ]; then
    echo "[Figure5MetricsParallel] missing decoded dir: ${infer_dir}" >&2
    exit 2
  fi

  cmd=(
    "${TORCHRUN}" --master_port="${master_port}" --nproc_per_node=1 -m eval.javisbench.main
    --input_file "${MANIFEST}"
    --infer_data_dir "${infer_dir}"
    --output_file "${output_file}"
    --metrics ${METRICS}
    --exclude ${EXCLUDE}
    --max_audio_len_s "${MAX_AUDIO_LEN_S}"
    --num_workers "${NUM_WORKERS}"
    --cavp_config_path "${JAVIS_DIR}/eval/javisbench/configs/Stage1_CAVP.yaml"
  )
  if [ "${FORCE_EVAL}" = "1" ]; then
    cmd+=(--force_eval)
  fi

  echo "[Figure5MetricsParallel] starting ${config_name} gpu=${gpu} output=${output_file}"
  (CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}") >"${log_file}" 2>&1 &
  pids+=("$!")
  configs+=("${config_name}")
done

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[Figure5MetricsParallel] ${configs[$i]} failed" >&2
    status=1
  fi
done
if [ "${status}" -ne 0 ]; then
  exit "${status}"
fi

echo "[Figure5MetricsParallel] done output=${RESULTS_ROOT}"
