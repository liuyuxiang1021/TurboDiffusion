#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/project/TurboDiffusion/turbot2va"
LTX_DIR="${ROOT_DIR}/LTX-2"
PYTHON="${PYTHON:-/root/miniconda3/envs/turbot2av/bin/python}"
LATENT_ROOT="${LATENT_ROOT:?Set LATENT_ROOT to the figure5_latents_full output directory}"
DECODE_ROOT="${DECODE_ROOT:-${LATENT_ROOT}_decoded}"
CONFIGS="${CONFIGS:-dense topk_0.5 topk_0.4 topk_0.3 topk_0.2}"
GPUS="${GPUS:-0 1 2 3}"
THREADS_PER_PROC="${THREADS_PER_PROC:-24}"
OVERWRITE="${OVERWRITE:-0}"
LOG_SUFFIX="${LOG_SUFFIX:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS_PER_PROC}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${THREADS_PER_PROC}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${THREADS_PER_PROC}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${THREADS_PER_PROC}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PYTHONPATH:-packages/ltx-distillation/src:packages/ltx-core/src:packages/ltx-pipelines/src}"
export TURBO_CHECKPOINT_PATH="${TURBO_CHECKPOINT_PATH:-/root/autodl-tmp/checkpoints/turbot2av/LTX-2/ltx-2-19b-dev.safetensors}"

CONFIG_PATH="${CONFIG_PATH:-packages/ltx-distillation/configs/bidirectional_rcm.yaml}"

read -r -a GPU_ARRAY <<< "${GPUS}"
NUM_SHARDS="${#GPU_ARRAY[@]}"

mkdir -p "${DECODE_ROOT}"
cd "${LTX_DIR}"

for config_name in ${CONFIGS}; do
  latent_dir="${LATENT_ROOT}/${config_name}/latents"
  output_dir="${DECODE_ROOT}/${config_name}"
  log_dir="${output_dir}/logs"
  mkdir -p "${log_dir}"

  if [ ! -d "${latent_dir}" ]; then
    echo "[Figure5Decode] missing latent dir: ${latent_dir}" >&2
    exit 2
  fi

  echo "[Figure5Decode] starting ${config_name}, latent=${latent_dir}, output=${output_dir}"
  pids=()
  for shard_id in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$shard_id]}"
    log_path="${log_dir}/shard_${shard_id}${LOG_SUFFIX}.log"
    timing_path="${output_dir}/decode_timing_shard_${shard_id}.json"
    decode_args=(
      --config_path "${CONFIG_PATH}"
      --latent_dir "${latent_dir}"
      --output_dir "${output_dir}"
      --num_shards "${NUM_SHARDS}"
      --shard_id "${shard_id}"
      --timing_json "${timing_path}"
    )
    if [ "${OVERWRITE}" = "1" ]; then
      decode_args+=(--overwrite)
    fi
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      "${PYTHON}" "${ROOT_DIR}/scripts/decode_figure5_latents.py" \
        "${decode_args[@]}"
    ) >"${log_path}" 2>&1 &
    pids+=("$!")
    echo "[Figure5Decode] ${config_name} shard=${shard_id}/${NUM_SHARDS} gpu=${gpu} pid=${pids[-1]} log=${log_path}"
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [ "${status}" -ne 0 ]; then
    echo "[Figure5Decode] ${config_name} failed; inspect ${log_dir}" >&2
    exit "${status}"
  fi
done

echo "[Figure5Decode] done output=${DECODE_ROOT}"
