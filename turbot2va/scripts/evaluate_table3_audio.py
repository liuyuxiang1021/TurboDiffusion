#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch


AXES = ("CE", "CU", "PC", "PQ")


def load_prompts(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            field = next((name for name in ("prompt", "caption", "text") if name in (reader.fieldnames or [])), None)
            if field is None:
                raise ValueError(f"No prompt/caption/text column in {path}")
            return [str(row[field]).strip() for row in reader if str(row[field]).strip()]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def locate_samples(decoded_dir: Path, prompts: list[str], num_samples: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(num_samples):
        stem = f"sample_{index:04d}"
        direct = decoded_dir / f"{stem}.wav"
        candidates = [direct] if direct.is_file() else sorted(decoded_dir.rglob(f"{stem}.wav"))
        if not candidates:
            raise FileNotFoundError(f"Missing {stem}.wav under {decoded_dir}")
        rows.append({"index": index, "sample": stem, "path": str(candidates[0]), "prompt": prompts[index]})
    return rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def evaluate_aes(rows: list[dict[str, Any]], output_path: Path, batch_size: int) -> list[dict[str, float]]:
    if output_path.is_file():
        cached = load_jsonl(output_path)
        if len(cached) == len(rows):
            return cached

    from audiobox_aesthetics.infer import initialize_predictor

    predictor = initialize_predictor()
    results: list[dict[str, float]] = []
    for start in range(0, len(rows), batch_size):
        batch = [{"path": row["path"]} for row in rows[start : start + batch_size]]
        results.extend(predictor.forward(batch))
        print(f"[Table3][AES] {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    write_jsonl(output_path, results)
    del predictor
    torch.cuda.empty_cache()
    return results


def evaluate_msclap(
    rows: list[dict[str, Any]],
    output_path: Path,
    batch_size: int,
    vendor_path: Path,
) -> list[dict[str, float]]:
    if output_path.is_file():
        cached = load_jsonl(output_path)
        if len(cached) == len(rows):
            return cached

    sys.path.insert(0, str(vendor_path))
    from msclap import CLAP

    model = CLAP(version="2023", use_cuda=torch.cuda.is_available())

    def get_text_embeddings(prompts: list[str]) -> torch.Tensor:
        # MS-CLAP's bundled wrapper pads but does not truncate long prompts,
        # which makes variable-length custom prompts impossible to collate.
        tokenized = []
        for prompt in prompts:
            if "gpt" in model.args.text_model:
                prompt = prompt + " <|endoftext|>"
            tokens = model.tokenizer.encode_plus(
                text=prompt,
                add_special_tokens=True,
                max_length=model.args.text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            for key in model.token_keys:
                tensor = tokens[key].reshape(-1)
                tokens[key] = tensor.cuda() if model.use_cuda and torch.cuda.is_available() else tensor
            tokenized.append(tokens)
        return model._get_text_embeddings(model.default_collate(tokenized))

    results: list[dict[str, float]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        audio_embeddings = model.get_audio_embeddings([row["path"] for row in batch], resample=True)
        text_embeddings = get_text_embeddings([row["prompt"] for row in batch])
        scores = torch.nn.functional.cosine_similarity(audio_embeddings, text_embeddings).detach().cpu().tolist()
        results.extend({"MS_CLAP": float(score)} for score in scores)
        print(f"[Table3][MS-CLAP] {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    write_jsonl(output_path, results)
    return results


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TurboT2AV Table III metrics on decoded audio")
    parser.add_argument("--decoded_dir", type=Path, required=True)
    parser.add_argument("--prompts_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--aes_batch_size", type=int, default=16)
    parser.add_argument("--clap_batch_size", type=int, default=16)
    parser.add_argument(
        "--msclap_vendor",
        type=Path,
        default=Path("/root/autodl-tmp/project/pixi-envs/ttabench/vendor_msclap"),
    )
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file)
    if len(prompts) < args.num_samples:
        raise ValueError(f"Need {args.num_samples} prompts, found {len(prompts)}")
    rows = locate_samples(args.decoded_dir, prompts, args.num_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "inputs.jsonl", rows)

    aes = evaluate_aes(rows, args.output_dir / "aes_per_sample.jsonl", args.aes_batch_size)
    clap = evaluate_msclap(rows, args.output_dir / "msclap_per_sample.jsonl", args.clap_batch_size, args.msclap_vendor)

    axis_means = {axis: mean([float(result[axis]) for result in aes]) for axis in AXES}
    summary = {
        "num_samples": len(rows),
        "content_enjoyment": axis_means["CE"],
        "content_usefulness": axis_means["CU"],
        "production_complexity": axis_means["PC"],
        "production_quality": axis_means["PQ"],
        "aes_mean": mean(list(axis_means.values())),
        "ms_clap": mean([float(result["MS_CLAP"]) for result in clap]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
