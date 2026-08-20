#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def load_records(input_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("timing_shard_*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for record in payload.get("records", []):
            row = dict(record)
            row["timing_file"] = str(path)
            records.append(row)
    return records


def summarize(config_name: str, records: list[dict[str, Any]], dense_mean: float | None = None) -> dict[str, Any]:
    times = [float(row["generator_seconds"]) for row in records]
    summary: dict[str, Any] = {
        "config": config_name,
        "num_records": len(times),
        "mean_generator_seconds": statistics.fmean(times) if times else None,
        "median_generator_seconds": statistics.median(times) if times else None,
        "min_generator_seconds": min(times) if times else None,
        "max_generator_seconds": max(times) if times else None,
        "stdev_generator_seconds": statistics.stdev(times) if len(times) > 1 else None,
    }
    if dense_mean and summary["mean_generator_seconds"]:
        summary["speedup_vs_dense"] = dense_mean / summary["mean_generator_seconds"]
    else:
        summary["speedup_vs_dense"] = 1.0 if config_name == "dense" and summary["mean_generator_seconds"] else None
    return summary


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "sample",
        "seed",
        "attention_type",
        "attention_scope",
        "sla_topk",
        "fast_norm",
        "trim_text_context",
        "quant_linear",
        "quant_linear_backend",
        "generator_seconds",
        "timing_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config",
        "num_records",
        "mean_generator_seconds",
        "median_generator_seconds",
        "min_generator_seconds",
        "max_generator_seconds",
        "stdev_generator_seconds",
        "speedup_vs_dense",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--config_name", default=None)
    parser.add_argument("--dense_summary", default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--all_configs", action="store_true", default=False)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    if args.all_configs:
        summaries = []
        dense_mean = None
        dense_path = input_dir / "dense" / "summary.json"
        if dense_path.exists():
            dense_mean = json.loads(dense_path.read_text(encoding="utf-8")).get("mean_generator_seconds")
        for summary_path in sorted(input_dir.glob("*/summary.json")):
            item = json.loads(summary_path.read_text(encoding="utf-8"))
            if dense_mean and item.get("mean_generator_seconds"):
                item["speedup_vs_dense"] = dense_mean / item["mean_generator_seconds"]
            summaries.append(item)
        output_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        write_summary_csv(output_csv, summaries)
        return

    records = load_records(input_dir)
    dense_mean = None
    if args.dense_summary and Path(args.dense_summary).exists():
        dense_mean = json.loads(Path(args.dense_summary).read_text(encoding="utf-8")).get("mean_generator_seconds")

    summary = summarize(args.config_name or input_dir.name, records, dense_mean=dense_mean)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_records_csv(output_csv, records)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
