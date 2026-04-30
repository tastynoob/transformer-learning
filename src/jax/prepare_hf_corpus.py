"""Prepare a local normalized JSONL corpus from Hugging Face datasets.

Default target is HuggingFaceTB/smollm-corpus, subset cosmopedia-v2. The output
schema is tl-corpus-v1: one JSON object per line with trainable text segments.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable
import unicodedata

try:
    from .corpus import TrainingSegment, make_record
except ImportError:
    from corpus import TrainingSegment, make_record


@dataclass
class HFCorpusConfig:
    dataset: str = "HuggingFaceTB/smollm-corpus"
    subset: str = "cosmopedia-v2"
    split: str = "train"
    text_field: str = "text"
    out_path: Path = Path("data/corpus/smollm_cosmopedia_v2.tl.jsonl")
    streaming: bool = True
    max_records: int | None = 50000
    min_chars: int = 200
    max_chars_per_record: int | None = 6000
    shuffle_buffer: int = 10000
    seed: int = 0
    log_every: int = 1000


HF_CORPUS_CFG = HFCorpusConfig()


def require_load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with `pip install datasets`, "
            "then run `python3 src/jax/prepare_hf_corpus.py` again."
        ) from exc
    return load_dataset


def jsonable_config(cfg: HFCorpusConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(coerce_text(item) for item in value if item is not None)
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def extract_text(record: dict[str, Any], text_field: str) -> str:
    if text_field not in record:
        available = ", ".join(sorted(record.keys()))
        raise KeyError(f"record does not contain text_field={text_field!r}; available fields: {available}")
    return coerce_text(record[text_field])


def iter_dataset(cfg: HFCorpusConfig) -> Iterable[dict[str, Any]]:
    load_dataset = require_load_dataset()
    dataset = load_dataset(
        cfg.dataset,
        cfg.subset,
        split=cfg.split,
        streaming=cfg.streaming,
    )
    if cfg.streaming and cfg.shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)
    elif not cfg.streaming:
        dataset = dataset.shuffle(seed=cfg.seed)
    return dataset


def write_corpus(cfg: HFCorpusConfig = HF_CORPUS_CFG) -> dict[str, Any]:
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cfg.out_path.with_name(cfg.out_path.name + ".tmp")
    meta_path = cfg.out_path.with_name(cfg.out_path.name + ".meta.json")

    stats: dict[str, Any] = {
        "dataset": cfg.dataset,
        "subset": cfg.subset,
        "split": cfg.split,
        "scanned_records": 0,
        "written_records": 0,
        "skipped_short_records": 0,
        "written_chars": 0,
    }

    start = time.time()
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in iter_dataset(cfg):
            stats["scanned_records"] += 1
            text = normalize_text(extract_text(record, cfg.text_field))
            if len(text) < cfg.min_chars:
                stats["skipped_short_records"] += 1
                continue
            if cfg.max_chars_per_record is not None and len(text) > cfg.max_chars_per_record:
                text = text[: cfg.max_chars_per_record].rstrip()

            payload = make_record(
                f"hf-{stats['written_records'] + 1:06d}",
                f"{cfg.dataset}/{cfg.subset}:{cfg.split}",
                [TrainingSegment("text", text, True)],
                {"source_record": stats["scanned_records"]},
            )
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stats["written_records"] += 1
            stats["written_chars"] += len(text)

            if cfg.log_every > 0 and stats["written_records"] % cfg.log_every == 0:
                elapsed = time.time() - start
                print(
                    "written="
                    f"{stats['written_records']} scanned={stats['scanned_records']} "
                    f"chars={stats['written_chars']} elapsed={elapsed:.1f}s"
                )

            if cfg.max_records is not None and stats["written_records"] >= cfg.max_records:
                break

    tmp_path.replace(cfg.out_path)
    stats["elapsed_sec"] = round(time.time() - start, 3)
    meta_path.write_text(
        json.dumps({"config": jsonable_config(cfg), "stats": stats}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved corpus: {cfg.out_path}")
    print(f"saved metadata: {meta_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a local JSONL corpus from Hugging Face datasets.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--max-chars-per-record", type=int, default=None)
    parser.add_argument("--shuffle-buffer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--no-streaming", action="store_true")
    return parser.parse_args()


def cfg_from_args(args: argparse.Namespace) -> HFCorpusConfig:
    updates: dict[str, Any] = {}
    for arg_name, field_name in (
        ("dataset", "dataset"),
        ("subset", "subset"),
        ("split", "split"),
        ("text_field", "text_field"),
        ("out_path", "out_path"),
        ("max_records", "max_records"),
        ("min_chars", "min_chars"),
        ("max_chars_per_record", "max_chars_per_record"),
        ("shuffle_buffer", "shuffle_buffer"),
        ("seed", "seed"),
        ("log_every", "log_every"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value
    if args.no_streaming:
        updates["streaming"] = False
    return replace(HF_CORPUS_CFG, **updates)


def main() -> None:
    write_corpus(cfg_from_args(parse_args()))
    sys.stdout.flush()
    sys.stderr.flush()
    # With proxychains, datasets/pyarrow can abort during interpreter teardown
    # after all files are already written. Avoid reporting that cleanup issue as
    # a failed corpus generation.
    os._exit(0)


if __name__ == "__main__":
    main()
