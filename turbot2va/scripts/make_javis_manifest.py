#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_prompts(path: Path, limit: int | None) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return prompts[:limit] if limit is not None else prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--num_prompts", type=int, default=None)
    args = parser.parse_args()

    prompts = load_prompts(Path(args.prompts_file), args.num_prompts)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "path", "audio_path"])
        writer.writeheader()
        for prompt in prompts:
            writer.writerow({"text": prompt, "path": "/dev/null", "audio_path": "/dev/null"})


if __name__ == "__main__":
    main()
