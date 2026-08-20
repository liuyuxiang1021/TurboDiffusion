#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp/project/TurboDiffusion/turbot2va"
LTX_DIR="${ROOT_DIR}/LTX-2"
PYTHON="${PYTHON:-/root/miniconda3/envs/turbot2av/bin/python}"
PROMPTS_FILE="${PROMPTS_FILE:-/root/autodl-tmp/project/00_mydata/prompts.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs/turbot2av/figure5_prompts_$(date +%Y%m%d_%H%M%S)}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-1}"
SEED="${SEED:-12345}"
GPUS="${GPUS:-0 1 2 3}"
CONFIGS="${CONFIGS:-dense topk_0.5 topk_0.4 topk_0.3 topk_0.2}"
THREADS_PER_PROC="${THREADS_PER_PROC:-24}"
DECODE="${DECODE:-0}"
SAVE_LATENTS="${SAVE_LATENTS:-0}"
MEASURE_STAGES="${MEASURE_STAGES:-0}"
ATTENTION_SCOPE="${ATTENTION_SCOPE:-video_self}"
if [ "${SLA_TOPK_SCHEDULE_TEMPLATE+x}" = "x" ]; then
  SLA_TOPK_SCHEDULE_TEMPLATE_VALUE="${SLA_TOPK_SCHEDULE_TEMPLATE}"
else
  SLA_TOPK_SCHEDULE_TEMPLATE_VALUE='24-47:{topk}'
fi
ENABLE_FAST_NORM="${ENABLE_FAST_NORM:-1}"
ENABLE_QUANT_LINEAR="${ENABLE_QUANT_LINEAR:-0}"
ENABLE_TRIM_TEXT_CONTEXT="${ENABLE_TRIM_TEXT_CONTEXT:-0}"
QUANT_LINEAR_SCOPE="${QUANT_LINEAR_SCOPE:-all}"
QUANT_LINEAR_BACKEND="${QUANT_LINEAR_BACKEND:-tilelang_postscale}"

export MAX_JOBS="${MAX_JOBS:-${THREADS_PER_PROC}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS_PER_PROC}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${THREADS_PER_PROC}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${THREADS_PER_PROC}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${THREADS_PER_PROC}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PYTHONPATH:-packages/ltx-distillation/src:packages/ltx-core/src:packages/ltx-pipelines/src}"
export TURBO_CHECKPOINT_PATH="${TURBO_CHECKPOINT_PATH:-/root/autodl-tmp/checkpoints/turbot2av/LTX-2/ltx-2-19b-dev.safetensors}"
export TURBO_GEMMA_PATH="${TURBO_GEMMA_PATH:-/root/autodl-tmp/checkpoints/turbot2av/gemma-3-12b-it-qat-q4_0-unquantized}"

STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-/root/autodl-tmp/checkpoints/turbot2av/turbo-t2av-weights/checkpoints/turbot2av_main/model.pth}"
CONFIG_PATH="${CONFIG_PATH:-packages/ltx-distillation/configs/bidirectional_rcm.yaml}"

read -r -a GPU_ARRAY <<< "${GPUS}"
NUM_SHARDS="${#GPU_ARRAY[@]}"

mkdir -p "${OUTPUT_ROOT}"
cd "${LTX_DIR}"

common_args=(
  --config_path "${CONFIG_PATH}"
  --prompts_file "${PROMPTS_FILE}"
  --model_kind student
  --student_checkpoint "${STUDENT_CHECKPOINT}"
  --student_param auto
  --no_init_lock
  --num_prompts "${NUM_PROMPTS}"
  --seed "${SEED}"
  --num_frames 121
  --video_height 1024
  --video_width 1792
  --warmup_samples "${WARMUP_SAMPLES}"
  --preencode_text
  --overwrite
  --num_shards "${NUM_SHARDS}"
)

if [ "${DECODE}" != "1" ]; then
  common_args+=(--skip_decode)
fi
if [ "${SAVE_LATENTS}" = "1" ]; then
  common_args+=(--save_latents)
fi
if [ "${MEASURE_STAGES}" = "1" ]; then
  common_args+=(--measure_stages)
fi

accel_args=(
  --attention_type sagesla
  --attention_scope "${ATTENTION_SCOPE}"
)
if [ "${ENABLE_FAST_NORM}" = "1" ]; then
  accel_args+=(--fast_norm)
fi
if [ "${ENABLE_QUANT_LINEAR}" = "1" ]; then
  accel_args+=(
    --quant_linear
    --quant_linear_scope "${QUANT_LINEAR_SCOPE}"
    --quant_linear_backend "${QUANT_LINEAR_BACKEND}"
  )
fi
if [ "${ENABLE_TRIM_TEXT_CONTEXT}" = "1" ]; then
  accel_args+=(--trim_text_context)
fi

config_args() {
  local topk
  local schedule
  case "$1" in
    dense)
      ;;
    topk_*)
      topk="${1#topk_}"
      if ! [[ "${topk}" =~ ^0([.][0-9]+)?$|^1([.]0+)?$ ]]; then
        echo "Invalid top-k config: $1" >&2
        return 2
      fi
      ;;
    *)
      echo "Unknown config: $1" >&2
      return 2
      ;;
  esac
  if [ -n "${topk:-}" ]; then
    printf '%s\n' "${accel_args[@]}"
    if [ -n "${SLA_TOPK_SCHEDULE_TEMPLATE_VALUE}" ]; then
      schedule="${SLA_TOPK_SCHEDULE_TEMPLATE_VALUE//\{topk\}/${topk}}"
      printf '%s\n' --sla_topk 1.0 --sla_topk_schedule "${schedule}"
    else
      printf '%s\n' --sla_topk "${topk}"
    fi
  fi
}

aggregate_config() {
  local config_name="$1"
  local config_dir="$2"
  "${PYTHON}" "${ROOT_DIR}/scripts/aggregate_figure5_timings.py" \
    --input_dir "${config_dir}" \
    --config_name "${config_name}" \
    --dense_summary "${OUTPUT_ROOT}/dense/summary.json" \
    --output_json "${config_dir}/summary.json" \
    --output_csv "${config_dir}/records.csv"
}

for config_name in ${CONFIGS}; do
  config_dir="${OUTPUT_ROOT}/${config_name}"
  log_dir="${config_dir}/logs"
  mkdir -p "${log_dir}"

  mapfile -t extra_args < <(config_args "${config_name}")
  echo "[Figure5Bench] starting ${config_name} on ${NUM_SHARDS} shard(s), output=${config_dir}"

  pids=()
  for shard_id in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$shard_id]}"
    log_path="${log_dir}/shard_${shard_id}.log"
    timing_path="${config_dir}/timing_shard_${shard_id}.json"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      "${PYTHON}" packages/ltx-distillation/src/ltx_distillation/tools/run_av_inference_eval.py \
        "${common_args[@]}" \
        --output_dir "${config_dir}" \
        --shard_id "${shard_id}" \
        --timing_json "${timing_path}" \
        "${extra_args[@]}"
    ) >"${log_path}" 2>&1 &
    pids+=("$!")
    echo "[Figure5Bench] ${config_name} shard=${shard_id}/${NUM_SHARDS} gpu=${gpu} pid=${pids[-1]} log=${log_path}"
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [ "${status}" -ne 0 ]; then
    echo "[Figure5Bench] ${config_name} failed; inspect ${log_dir}" >&2
    exit "${status}"
  fi
  aggregate_config "${config_name}" "${config_dir}"
done

"${PYTHON}" "${ROOT_DIR}/scripts/aggregate_figure5_timings.py" \
  --input_dir "${OUTPUT_ROOT}" \
  --all_configs \
  --output_json "${OUTPUT_ROOT}/summary.json" \
  --output_csv "${OUTPUT_ROOT}/summary.csv"

echo "[Figure5Bench] done output=${OUTPUT_ROOT}"
