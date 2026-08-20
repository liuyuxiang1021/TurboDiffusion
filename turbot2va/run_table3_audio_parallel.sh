#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/project/TurboDiffusion/turbot2va"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
DECODE_ROOT="${DECODE_ROOT:?Set DECODE_ROOT to the decoded student output directory}"
RESULTS_ROOT="${RESULTS_ROOT:-${DECODE_ROOT}_table3_audio}"
PROMPTS_FILE="${PROMPTS_FILE:-/root/autodl-tmp/project/00_mydata/prompts.txt}"
CONFIGS="${CONFIGS:-dense topk_0.5 topk_0.4 topk_0.3 topk_0.2}"
GPUS="${GPUS:-0 1 2 3}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
FORCE_EVAL="${FORCE_EVAL:-0}"

read -r -a gpu_array <<< "${GPUS}"
read -r -a config_array <<< "${CONFIGS}"
if [ "${#gpu_array[@]}" -eq 0 ]; then
  echo "No GPUs configured" >&2
  exit 2
fi

mkdir -p "${RESULTS_ROOT}"
pids=()
for gpu_idx in "${!gpu_array[@]}"; do
  gpu="${gpu_array[$gpu_idx]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    for ((config_idx = gpu_idx; config_idx < ${#config_array[@]}; config_idx += ${#gpu_array[@]})); do
      config_name="${config_array[$config_idx]}"
      decoded_dir="${DECODE_ROOT}/${config_name}"
      output_dir="${RESULTS_ROOT}/${config_name}"
      mkdir -p "${output_dir}"
      if [ "${FORCE_EVAL}" != "1" ] && [ -s "${output_dir}/summary.json" ]; then
        echo "[Table3] skip completed ${config_name}"
        continue
      fi
      echo "[Table3] start ${config_name} gpu=${gpu}"
      "${PYTHON}" "${ROOT_DIR}/scripts/evaluate_table3_audio.py" \
        --decoded_dir "${decoded_dir}" \
        --prompts_file "${PROMPTS_FILE}" \
        --output_dir "${output_dir}" \
        --num_samples "${NUM_SAMPLES}" \
        >"${output_dir}/table3.log" 2>&1
    done
  ) &
  pids+=("$!")
done

status=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    echo "[Table3] worker on GPU ${gpu_array[$idx]} failed; inspect ${RESULTS_ROOT}/*/table3.log" >&2
    status=1
  fi
done
exit "${status}"
