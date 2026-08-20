#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/project/TurboDiffusion/turbot2va"
LTX_DIR="${ROOT_DIR}/LTX-2"
PYTHON="${PYTHON:-/root/miniconda3/envs/turbot2av/bin/python}"
PROMPTS_FILE="${PROMPTS_FILE:-/root/autodl-tmp/project/00_mydata/prompts.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to the figure5 latent output directory}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-1}"
SEED="${SEED:-12345}"
GPUS="${GPUS:-0 1 2 3}"
THREADS_PER_PROC="${THREADS_PER_PROC:-24}"
TEACHER_MODE="${TEACHER_MODE:-native_rf}"
TEACHER_STEPS="${TEACHER_STEPS:-40}"
CONFIG_NAME="${CONFIG_NAME:-teacher}"
OVERWRITE="${OVERWRITE:-0}"

export MAX_JOBS="${MAX_JOBS:-${THREADS_PER_PROC}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS_PER_PROC}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${THREADS_PER_PROC}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${THREADS_PER_PROC}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${THREADS_PER_PROC}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PYTHONPATH:-packages/ltx-distillation/src:packages/ltx-core/src:packages/ltx-pipelines/src}"
export TURBO_CHECKPOINT_PATH="${TURBO_CHECKPOINT_PATH:-/root/autodl-tmp/checkpoints/turbot2av/LTX-2/ltx-2-19b-dev.safetensors}"
export TURBO_GEMMA_PATH="${TURBO_GEMMA_PATH:-/root/autodl-tmp/checkpoints/turbot2av/gemma-3-12b-it-qat-q4_0-unquantized}"

CONFIG_PATH="${CONFIG_PATH:-packages/ltx-distillation/configs/bidirectional_rcm.yaml}"

read -r -a GPU_ARRAY <<< "${GPUS}"
NUM_SHARDS="${#GPU_ARRAY[@]}"
CONFIG_DIR="${OUTPUT_ROOT}/${CONFIG_NAME}"
LOG_DIR="${CONFIG_DIR}/logs"

mkdir -p "${LOG_DIR}"
cd "${LTX_DIR}"

echo "[Figure5Teacher] starting ${CONFIG_NAME} mode=${TEACHER_MODE} steps=${TEACHER_STEPS} shards=${NUM_SHARDS} output=${CONFIG_DIR}"

pids=()
for shard_id in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$shard_id]}"
  log_path="${LOG_DIR}/shard_${shard_id}.log"
  timing_path="${CONFIG_DIR}/timing_shard_${shard_id}.json"
  overwrite_args=()
  if [ "${OVERWRITE}" = "1" ]; then
    overwrite_args+=(--overwrite)
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON}" packages/ltx-distillation/src/ltx_distillation/tools/run_av_inference_eval.py \
      --config_path "${CONFIG_PATH}" \
      --prompts_file "${PROMPTS_FILE}" \
      --output_dir "${CONFIG_DIR}" \
      --model_kind teacher \
      --teacher_mode "${TEACHER_MODE}" \
      --teacher_steps "${TEACHER_STEPS}" \
      --no_init_lock \
      --num_prompts "${NUM_PROMPTS}" \
      --seed "${SEED}" \
      --num_frames 121 \
      --video_height 1024 \
      --video_width 1792 \
      --warmup_samples "${WARMUP_SAMPLES}" \
      --skip_decode \
      --save_latents \
      --preencode_text \
      --num_shards "${NUM_SHARDS}" \
      --shard_id "${shard_id}" \
      --timing_json "${timing_path}" \
      "${overwrite_args[@]}"
  ) >"${log_path}" 2>&1 &
  pids+=("$!")
  echo "[Figure5Teacher] shard=${shard_id}/${NUM_SHARDS} gpu=${gpu} pid=${pids[-1]} log=${log_path}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [ "${status}" -ne 0 ]; then
  echo "[Figure5Teacher] failed; inspect ${LOG_DIR}" >&2
  exit "${status}"
fi

"${PYTHON}" "${ROOT_DIR}/scripts/aggregate_figure5_timings.py" \
  --input_dir "${CONFIG_DIR}" \
  --config_name "${CONFIG_NAME}" \
  --dense_summary "${OUTPUT_ROOT}/dense/summary.json" \
  --output_json "${CONFIG_DIR}/summary.json" \
  --output_csv "${CONFIG_DIR}/records.csv"

echo "[Figure5Teacher] done output=${CONFIG_DIR}"
