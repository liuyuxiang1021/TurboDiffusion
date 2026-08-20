#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


CONFIGS = ("dense", "topk_0.5", "topk_0.4", "topk_0.3", "topk_0.2")
VBENCH_DIMENSIONS = (
    "aesthetic_quality",
    "imaging_quality",
    "motion_smoothness",
    "subject_consistency",
    "temporal_flickering",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def scalar(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("overall")
    if isinstance(value, list):
        return value[0] if value else None
    return value


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_decode_mean(path: Path | None) -> float | None:
    if path is None or not path.is_file():
        return None
    records = load_json(path).get("records", [])
    values = [float(record["decode_seconds"]) for record in records if "decode_seconds" in record]
    return statistics.fmean(values) if values else None


def load_vbench_metric(root: Path, config: str, dimension: str) -> float | None:
    candidates = sorted((root / config / dimension).glob("*_eval_results.json"))
    if not candidates:
        return None
    result = load_json(candidates[-1]).get(dimension)
    value = scalar(result)
    return float(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine TurboT2VA student results into paper-style tables")
    parser.add_argument("--latent_root", type=Path, required=True)
    parser.add_argument("--javis_root", type=Path, required=True)
    parser.add_argument(
        "--javis_supplement_root",
        type=Path,
        action="append",
        default=[],
        help="Optional metric directories to merge in command-line order; may be repeated.",
    )
    parser.add_argument("--vbench_root", type=Path, required=True)
    parser.add_argument("--table3_root", type=Path, required=True)
    parser.add_argument("--decode_timing", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--vbench_scale", type=float, default=100.0)
    args = parser.parse_args()

    decode_seconds = load_decode_mean(args.decode_timing)
    timing_by_config = {
        config: load_json(args.latent_root / config / "summary.json") for config in args.configs
    }
    dense_generator = timing_by_config.get("dense", {}).get("mean_generator_seconds")
    dense_total = float(dense_generator) + decode_seconds if dense_generator is not None and decode_seconds is not None else None

    table1: list[dict[str, Any]] = []
    table2: list[dict[str, Any]] = []
    table3: list[dict[str, Any]] = []
    for config in args.configs:
        timing = timing_by_config[config]
        metrics = load_json(args.javis_root / f"{config}.json")
        for supplement_root in args.javis_supplement_root:
            metrics.update(load_json(supplement_root / f"{config}.json"))
        audio_metrics = load_json(args.table3_root / config / "summary.json")

        generator_seconds = timing.get("mean_generator_seconds")
        total_seconds = (
            float(generator_seconds) + decode_seconds
            if generator_seconds is not None and decode_seconds is not None
            else None
        )
        table1.append(
            {
                "config": config,
                "num_samples": timing.get("num_records"),
                "generator_seconds": generator_seconds,
                "generator_speedup": (
                    float(dense_generator) / float(generator_seconds)
                    if dense_generator is not None and generator_seconds
                    else None
                ),
                "decode_seconds_10_sample_mean": decode_seconds,
                "estimated_total_seconds": total_seconds,
                "estimated_total_speedup": dense_total / total_seconds if dense_total and total_seconds else None,
                "visual_quality": scalar(metrics.get("visual_quality")),
                "motion_quality": scalar(metrics.get("motion_quality")),
                "audio_quality": audio_metrics.get("aes_mean"),
                "ib_tv": scalar(metrics.get("ib_tv")),
                "ib_ta": scalar(metrics.get("ib_ta")),
                "clip": scalar(metrics.get("clip_score")),
                "clap": scalar(metrics.get("clap_score")),
                "ib_av": scalar(metrics.get("ib_av")),
                "cavp": scalar(metrics.get("cavp_score")),
                "avh": scalar(metrics.get("avh_score")),
                "javis": scalar(metrics.get("javis_score")),
                "desync": scalar(metrics.get("desync")),
            }
        )

        table2_row: dict[str, Any] = {"config": config}
        for dimension in VBENCH_DIMENSIONS:
            value = load_vbench_metric(args.vbench_root, config, dimension)
            table2_row[dimension] = value * args.vbench_scale if value is not None else None
        table2.append(table2_row)

        table3.append(
            {
                "config": config,
                "num_samples": audio_metrics.get("num_samples"),
                "content_enjoyment": audio_metrics.get("content_enjoyment"),
                "content_usefulness": audio_metrics.get("content_usefulness"),
                "production_complexity": audio_metrics.get("production_complexity"),
                "production_quality": audio_metrics.get("production_quality"),
                "aes_mean": audio_metrics.get("aes_mean"),
                "ms_clap": audio_metrics.get("ms_clap"),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(args.output_dir / "table1.csv", table1)
    write_table(args.output_dir / "table2.csv", table2)
    write_table(args.output_dir / "table3.csv", table3)
    (args.output_dir / "all_results.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "vbench_scale": args.vbench_scale,
                    "decode_timing_samples": 10 if decode_seconds is not None else 0,
                    "estimated_total_seconds": "generator mean + shared 10-sample decode mean",
                },
                "table1": table1,
                "table2": table2,
                "table3": table3,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
