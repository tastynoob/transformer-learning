"""Prepare local normalized JSONL SFT/chat corpora from Hugging Face datasets.

Default target is HuggingFaceH4/ultrachat_200k train_sft. The output schema
is tl-corpus-v1: each row contains role segments and assistant-only train flags.
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
    from .corpus import TrainingSegment, make_record, parse_legacy_chat_text
except ImportError:
    from corpus import TrainingSegment, make_record, parse_legacy_chat_text


@dataclass
class SFTCorpusConfig:
    dataset: str = "HuggingFaceH4/ultrachat_200k"
    subset: str | None = None
    split: str = "train_sft"
    out_path: Path = Path("data/corpus/ultrachat_200k_chat.tl.jsonl")
    format: str = "messages"  # messages, conversations, instruction, daily_dialog
    messages_field: str = "messages"
    dialog_field: str = "dialog"
    instruction_field: str = "instruction"
    input_field: str = "input"
    output_field: str = "output"
    streaming: bool = True
    max_records: int | None = 50000
    min_chars: int = 80
    max_chars_per_record: int | None = 6000
    shuffle_buffer: int = 10000
    seed: int = 0
    log_every: int = 1000
    user_label: str = "User"
    assistant_label: str = "Assistant"
    role_separator: str = ": "


SFT_CORPUS_CFG = SFTCorpusConfig()


def require_load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with `pip install datasets`, "
            "then run `python3 src/jax/prepare_sft_corpus.py` again."
        ) from exc
    return load_dataset


def jsonable_config(cfg: SFTCorpusConfig) -> dict[str, Any]:
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


def role_label(role: str, cfg: SFTCorpusConfig) -> str | None:
    role = role.strip().lower()
    if role in {"user", "human", "prompter"}:
        return cfg.user_label
    if role in {"assistant", "gpt", "bot"}:
        return cfg.assistant_label
    return None


def message_role(message: dict[str, Any]) -> str:
    return coerce_text(message.get("role", message.get("from", "")))


def message_content(message: dict[str, Any]) -> str:
    return coerce_text(message.get("content", message.get("value", "")))


def append_turn(parts: list[str], label: str, content: str, cfg: SFTCorpusConfig) -> bool:
    content = normalize_text(content)
    if not content:
        return False
    turn = f"{label}{cfg.role_separator}{content}"
    candidate = "\n".join(parts + [turn])
    if cfg.max_chars_per_record is not None and len(candidate) > cfg.max_chars_per_record:
        return False
    parts.append(turn)
    return True


def format_message_list(messages: Any, cfg: SFTCorpusConfig) -> str:
    if not isinstance(messages, list):
        return ""

    parts: list[str] = []
    assistant_turns = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        label = role_label(message_role(message), cfg)
        if label is None:
            continue
        if not append_turn(parts, label, message_content(message), cfg):
            break
        assistant_turns += int(label == cfg.assistant_label)

    if assistant_turns == 0 or len(parts) < 2:
        return ""
    return "\n".join(parts)


def format_instruction_record(record: dict[str, Any], cfg: SFTCorpusConfig) -> str:
    instruction = normalize_text(coerce_text(record.get(cfg.instruction_field, "")))
    input_text = normalize_text(coerce_text(record.get(cfg.input_field, "")))
    output = normalize_text(coerce_text(record.get(cfg.output_field, "")))
    if not instruction or not output:
        return ""

    user_text = instruction if not input_text else f"{instruction}\n{input_text}"
    text = f"{cfg.user_label}{cfg.role_separator}{user_text}\n{cfg.assistant_label}{cfg.role_separator}{output}"
    if cfg.max_chars_per_record is not None and len(text) > cfg.max_chars_per_record:
        return ""
    return text


def format_daily_dialog_record(record: dict[str, Any], cfg: SFTCorpusConfig) -> list[TrainingSegment]:
    turns = record.get(cfg.dialog_field)
    if not isinstance(turns, list):
        return []

    segments: list[TrainingSegment] = []
    total_chars = 0
    for index, turn in enumerate(turns):
        content = normalize_text(coerce_text(turn))
        if not content:
            continue
        candidate_chars = total_chars + len(content)
        if cfg.max_chars_per_record is not None and candidate_chars > cfg.max_chars_per_record:
            break
        role = "user" if index % 2 == 0 else "assistant"
        segments.append(TrainingSegment(role, content, role == "assistant"))
        total_chars = candidate_chars

    if len(segments) < 2 or not any(segment.role == "assistant" for segment in segments):
        return []
    return segments


def format_record(record: dict[str, Any], cfg: SFTCorpusConfig) -> str:
    fmt = cfg.format.lower()
    if fmt == "messages":
        return format_message_list(record.get(cfg.messages_field), cfg)
    if fmt == "conversations":
        return format_message_list(record.get("conversations"), cfg)
    if fmt == "instruction":
        return format_instruction_record(record, cfg)
    raise ValueError(f"unsupported format={cfg.format!r}; expected messages, conversations, instruction, or daily_dialog")


def format_record_segments(record: dict[str, Any], cfg: SFTCorpusConfig) -> list[TrainingSegment]:
    if cfg.format.lower() == "daily_dialog":
        return format_daily_dialog_record(record, cfg)
    text = normalize_text(format_record(record, cfg))
    if not text:
        return []
    return parse_legacy_chat_text(text)


def iter_dataset(cfg: SFTCorpusConfig) -> Iterable[dict[str, Any]]:
    load_dataset = require_load_dataset()
    args = [cfg.dataset]
    if cfg.subset:
        args.append(cfg.subset)
    dataset = load_dataset(*args, split=cfg.split, streaming=cfg.streaming)
    if cfg.streaming and cfg.shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)
    elif not cfg.streaming:
        dataset = dataset.shuffle(seed=cfg.seed)
    return dataset


def write_corpus(cfg: SFTCorpusConfig = SFT_CORPUS_CFG) -> dict[str, Any]:
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cfg.out_path.with_name(cfg.out_path.name + ".tmp")
    meta_path = cfg.out_path.with_name(cfg.out_path.name + ".meta.json")

    stats: dict[str, Any] = {
        "dataset": cfg.dataset,
        "subset": cfg.subset,
        "split": cfg.split,
        "format": cfg.format,
        "scanned_records": 0,
        "written_records": 0,
        "skipped_records": 0,
        "written_chars": 0,
    }

    start = time.time()
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in iter_dataset(cfg):
            stats["scanned_records"] += 1
            segments = format_record_segments(record, cfg)
            content_chars = sum(len(segment.content) for segment in segments)
            if content_chars < cfg.min_chars:
                stats["skipped_records"] += 1
                continue

            if not segments:
                stats["skipped_records"] += 1
                continue
            payload = make_record(
                f"sft-{stats['written_records'] + 1:06d}",
                f"{cfg.dataset}:{cfg.split}",
                list(segments),
                {"source_record": stats["scanned_records"]},
            )
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stats["written_records"] += 1
            stats["written_chars"] += content_chars

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
    parser = argparse.ArgumentParser(description="Prepare a local JSONL SFT/chat corpus from Hugging Face datasets.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--format", choices=("messages", "conversations", "instruction", "daily_dialog"), default=None)
    parser.add_argument("--messages-field", default=None)
    parser.add_argument("--dialog-field", default=None)
    parser.add_argument("--instruction-field", default=None)
    parser.add_argument("--input-field", default=None)
    parser.add_argument("--output-field", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--max-chars-per-record", type=int, default=None)
    parser.add_argument("--shuffle-buffer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--user-label", default=None)
    parser.add_argument("--assistant-label", default=None)
    parser.add_argument("--role-separator", default=None)
    parser.add_argument("--no-streaming", action="store_true")
    return parser.parse_args()


def cfg_from_args(args: argparse.Namespace) -> SFTCorpusConfig:
    updates: dict[str, Any] = {}
    for arg_name, field_name in (
        ("dataset", "dataset"),
        ("subset", "subset"),
        ("split", "split"),
        ("out_path", "out_path"),
        ("format", "format"),
        ("messages_field", "messages_field"),
        ("dialog_field", "dialog_field"),
        ("instruction_field", "instruction_field"),
        ("input_field", "input_field"),
        ("output_field", "output_field"),
        ("max_records", "max_records"),
        ("min_chars", "min_chars"),
        ("max_chars_per_record", "max_chars_per_record"),
        ("shuffle_buffer", "shuffle_buffer"),
        ("seed", "seed"),
        ("log_every", "log_every"),
        ("user_label", "user_label"),
        ("assistant_label", "assistant_label"),
        ("role_separator", "role_separator"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value
    if args.no_streaming:
        updates["streaming"] = False
    return replace(SFT_CORPUS_CFG, **updates)


def main() -> None:
    write_corpus(cfg_from_args(parse_args()))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
