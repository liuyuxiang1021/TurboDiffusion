#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from ltx_core.loader.registry import DummyRegistry
from ltx_distillation.models.vae_wrapper import create_vae_wrappers
from ltx_distillation.tools.run_av_inference_eval import _decode_and_save_sample


def complete_output(mp4_path: Path, wav_path: Path, json_path: Path) -> bool:
    return (
        mp4_path.exists()
        and wav_path.exists()
        and json_path.exists()
        and mp4_path.stat().st_size > 1024
        and wav_path.stat().st_size > 1024
        and json_path.stat().st_size > 0
    )


def selected(paths: list[Path], num_shards: int, shard_id: int) -> list[Path]:
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard_id must be in [0, num_shards)")
    return [path for idx, path in enumerate(paths) if idx % num_shards == shard_id]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--latent_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--timing_json", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--measure_stages", action="store_true", default=False)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    for key, env in [
        ("checkpoint_path", "TURBO_CHECKPOINT_PATH"),
        ("gemma_path", "TURBO_GEMMA_PATH"),
    ]:
        if os.environ.get(env):
            cfg[key] = os.environ[env]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if bool(getattr(cfg, "mixed_precision", True)) else torch.float32

    latent_paths = sorted(Path(args.latent_dir).glob("sample_*.pt"))
    shard_paths = selected(latent_paths, args.num_shards, args.shard_id)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max_samples must be >= 1")
        shard_paths = shard_paths[: args.max_samples]
    os.makedirs(args.output_dir, exist_ok=True)
    print(
        f"[DecodeLatents] latents={len(latent_paths)} shard={args.shard_id}/{args.num_shards} "
        f"selected={len(shard_paths)} output={args.output_dir}",
        flush=True,
    )

    registry = DummyRegistry()
    video_vae, audio_vae = create_vae_wrappers(
        checkpoint_path=cfg.checkpoint_path,
        device=device,
        dtype=dtype,
        registry=registry,
    )
    video_vae.eval()
    audio_vae.eval()

    records: list[dict[str, Any]] = []
    for local_idx, latent_path in enumerate(shard_paths, start=1):
        load_start = time.perf_counter()
        payload = torch.load(latent_path, map_location="cpu")
        latent_load_seconds = time.perf_counter() - load_start
        sample_stem = str(payload.get("sample") or latent_path.stem)
        mp4_path = Path(args.output_dir) / f"{sample_stem}.mp4"
        wav_path = Path(args.output_dir) / f"{sample_stem}.wav"
        json_path = Path(args.output_dir) / f"{sample_stem}.json"
        if not args.overwrite and complete_output(mp4_path, wav_path, json_path):
            print(f"[DecodeLatents] skip existing {sample_stem} {local_idx}/{len(shard_paths)}", flush=True)
            continue

        if args.measure_stages:
            torch.cuda.synchronize(device)
            h2d_start = time.perf_counter()
        video_latent = payload["video_latent"].to(device=device, dtype=dtype)
        audio_latent = payload["audio_latent"].to(device=device, dtype=dtype)
        torch.cuda.synchronize(device)
        latent_h2d_seconds = time.perf_counter() - h2d_start if args.measure_stages else None
        start = time.perf_counter()
        stage_timings = _decode_and_save_sample(
            video_vae=video_vae,
            audio_vae=audio_vae,
            video_latent=video_latent,
            audio_latent=audio_latent,
            prompt_idx=int(payload["index"]),
            prompt=str(payload["prompt"]),
            sample_stem=sample_stem,
            seed=int(payload["seed"]),
            seed_idx=int(payload.get("seed_idx", 0)),
            output_dir=args.output_dir,
            video_fps=int(getattr(cfg, "benchmark_video_fps", 24)),
            audio_sample_rate=int(getattr(cfg, "benchmark_audio_sample_rate", 24000)),
            measure_stages=args.measure_stages,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        record = {
            "sample": sample_stem,
            "latent_path": str(latent_path),
            "mp4": str(mp4_path),
            "wav": str(wav_path),
            "decode_seconds": elapsed,
        }
        if args.measure_stages:
            record["latent_load_seconds"] = latent_load_seconds
            record["latent_h2d_seconds"] = latent_h2d_seconds
            record.update(stage_timings)
        records.append(record)
        del video_latent, audio_latent, payload
        torch.cuda.empty_cache()
        print(
            f"[DecodeLatents] saved {sample_stem} {local_idx}/{len(shard_paths)} "
            f"decode={elapsed:.2f}s",
            flush=True,
        )

    if args.timing_json:
        Path(args.timing_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.timing_json, "w", encoding="utf-8") as f:
            json.dump({"num_records": len(records), "records": records}, f, indent=2)
        print(f"[DecodeLatents] wrote {args.timing_json}", flush=True)


if __name__ == "__main__":
    main()
