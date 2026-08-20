#!/usr/bin/env bash
set -euo pipefail

VBENCH_DIR="${VBENCH_DIR:-/root/autodl-tmp/project/VBench}"
PYTHON="${PYTHON:-/root/miniconda3/envs/vbench/bin/python}"
DECODE_ROOT="${DECODE_ROOT:?Set DECODE_ROOT to the decoded student output directory}"
RESULTS_ROOT="${RESULTS_ROOT:-${DECODE_ROOT}_vbench}"
CONFIGS="${CONFIGS:-dense topk_0.5 topk_0.4 topk_0.3 topk_0.2}"
GPUS="${GPUS:-0 1 2 3}"
DIMENSIONS="${DIMENSIONS:-aesthetic_quality imaging_quality motion_smoothness subject_consistency temporal_flickering}"

export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/root/autodl-tmp/checkpoints/vbench}"
export PYTHONPATH="${VBENCH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

read -r -a gpu_array <<< "${GPUS}"
read -r -a config_array <<< "${CONFIGS}"
read -r -a dimension_array <<< "${DIMENSIONS}"
if [ "${#gpu_array[@]}" -eq 0 ]; then
  echo "No GPUs configured" >&2
  exit 2
fi

mkdir -p "${RESULTS_ROOT}"
cd "${VBENCH_DIR}"

pids=()
for gpu_idx in "${!gpu_array[@]}"; do
  gpu="${gpu_array[$gpu_idx]}"
  (
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="$(( ${MASTER_PORT_BASE:-29600} + gpu ))"
    for ((config_idx = gpu_idx; config_idx < ${#config_array[@]}; config_idx += ${#gpu_array[@]})); do
      config_name="${config_array[$config_idx]}"
      config_dir="$(readlink -f "${DECODE_ROOT}/${config_name}")"
      videos_dir="${config_dir}/video"
      if ! find "${videos_dir}" -maxdepth 1 -name '*.mp4' -print -quit 2>/dev/null | grep -q .; then
        videos_dir="${config_dir}"
      fi
      count="$(find "${videos_dir}" -maxdepth 1 -name '*.mp4' | wc -l)"
      if [ "${count}" -eq 0 ]; then
        echo "No videos found for ${config_name}: ${videos_dir}" >&2
        exit 2
      fi

      output_dir="${RESULTS_ROOT}/${config_name}"
      mkdir -p "${output_dir}"
      echo "[Table2] start ${config_name} videos=${count} gpu=${gpu}"
      export CUDA_VISIBLE_DEVICES="${gpu}"
      for dimension in "${dimension_array[@]}"; do
        dimension_dir="${output_dir}/${dimension}"
        mkdir -p "${dimension_dir}"
        if find "${dimension_dir}" -maxdepth 1 -name '*_eval_results.json' -size +0c -print -quit | grep -q .; then
          echo "[Table2] skip completed ${config_name}/${dimension}"
          continue
        fi
        echo "[Table2] evaluate ${config_name}/${dimension}"
        "${PYTHON}" evaluate.py \
          --videos_path "${videos_dir}" \
          --dimension "${dimension}" \
          --mode custom_input \
          --output_path "${dimension_dir}" \
          >"${dimension_dir}/table2.log" 2>&1
      done
    done
  ) &
  pids+=("$!")
done

status=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    echo "[Table2] worker on GPU ${gpu_array[$idx]} failed; inspect ${RESULTS_ROOT}/*/table2.log" >&2
    status=1
  fi
done
exit "${status}"
