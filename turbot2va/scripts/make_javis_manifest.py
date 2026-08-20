#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


AUDIO_TERMS = re.compile(
    r"\b(?:audio|sound|music|voice|vocal|sing|sings|singing|sung|speak|speaks|"
    r"speaking|speech|say|says|dialogue|whisper|shout|footstep|chirp|bark|roar|"
    r"hum|buzz|rustl|crackl|clap|thud|rumble|squeal|coo|moo|meow|purr|"
    r"neigh|bleat|honk|chime|ring|knock|creak|splash|patter|hiss|whistle|"
    r"laugh|applause|siren|engine|tire|melody|instrument|guitar|piano|drum|"
    r"bass|microphone|heard|audible|noise)\w*\b",
    re.IGNORECASE,
)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def load_prompts(path: Path, limit: int | None) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return prompts[:limit] if limit is not None else prompts


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words])


def modality_texts(prompt: str, max_words: int) -> tuple[str, str]:
    sentences = [item.strip() for item in SENTENCE_BOUNDARY.split(prompt) if item.strip()]
    audio_sentences = [item for item in sentences if AUDIO_TERMS.search(item)]
    visual_sentences = [item for item in sentences if not AUDIO_TERMS.search(item)]

    # Long T2AV prompts usually introduce the visual scene first and audio later.
    # Keep the extraction deterministic and fall back to opposite ends of the prompt.
    video_text = " ".join(visual_sentences) if visual_sentences else prompt
    audio_text = " ".join(audio_sentences) if audio_sentences else " ".join(sentences[-2:])
    return truncate_words(video_text, max_words), truncate_words(audio_text, max_words)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument(
        "--modality_text",
        action="store_true",
        help="Add short video_text/audio_text captions for fixed-context text encoders.",
    )
    parser.add_argument("--modality_text_max_words", type=int, default=30)
    args = parser.parse_args()

    prompts = load_prompts(Path(args.prompts_file), args.num_prompts)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["text", "path", "audio_path"]
        if args.modality_text:
            fieldnames += ["video_text", "audio_text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for prompt in prompts:
            row = {"text": prompt, "path": "/dev/null", "audio_path": "/dev/null"}
            if args.modality_text:
                row["video_text"], row["audio_text"] = modality_texts(
                    prompt, args.modality_text_max_words
                )
            writer.writerow(row)


if __name__ == "__main__":
    main()
