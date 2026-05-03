"""Build a mixed short-chat SFT corpus in tl-corpus-v1 format.

The default mixture is meant to be a larger replacement for DailyDialog-only
training while keeping conversations short enough for small local models:

- HuggingFaceH4/ultrachat_200k train_sft: broad assistant chat
- OpenAssistant/oasst1: human-written assistant-style trees
- Estwld/empathetic_dialogues_llm: short emotional conversations
- local DailyDialog tl-corpus-v1, if available
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable
import unicodedata

try:
    from .corpus import TrainingSegment, coerce_text, make_record, normalize_role, record_from_payload
except ImportError:
    from corpus import TrainingSegment, coerce_text, make_record, normalize_role, record_from_payload


BAD_ASSISTANT_PHRASES = (
    "as an ai language model",
    "as a language model",
    "i do not have emotions",
    "i don't have emotions",
    "i do not have personal opinions",
    "i don't have personal opinions",
    "i cannot provide a personal opinion",
    "i am not capable of having feelings",
)


@dataclass
class ChatMixConfig:
    out_path: Path = Path("data/corpus/chat_mix_sft.tl.jsonl")
    seed: int = 0
    streaming: bool = True
    shuffle_buffer: int = 10000
    log_every: int = 1000

    ultrachat_dataset: str = "HuggingFaceH4/ultrachat_200k"
    ultrachat_split: str = "train_sft"
    ultrachat_records: int = 20000

    oasst_dataset: str = "OpenAssistant/oasst1"
    oasst_splits: str = "train,validation"
    oasst_records: int = 10000

    empathetic_dataset: str = "Estwld/empathetic_dialogues_llm"
    empathetic_split: str = "train"
    empathetic_records: int = 8000

    daily_dialog_local: Path = Path("data/corpus/daily_dialog_train.tl.jsonl")
    daily_dialog_dataset: str = "OpenRL/daily_dialog"
    daily_dialog_split: str = "train"
    daily_dialog_records: int = 5000

    min_turns: int = 2
    max_turns: int = 12
    min_record_chars: int = 80
    max_record_chars: int = 4500
    max_user_chars: int = 1500
    min_assistant_chars: int = 20
    max_assistant_chars: int = 2200
    english_ratio: float = 0.75
    max_toxicity: float = 0.5


CHAT_MIX_CFG = ChatMixConfig()


def require_load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with `pip install datasets`, "
            "then run `python3 src/jax/prepare_chat_mix_corpus.py` again."
        ) from exc
    return load_dataset


def jsonable_config(cfg: ChatMixConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def clean_content(value: Any) -> str:
    text = normalize_text(coerce_text(value))
    text = text.replace("_comma_", ",")
    return text.strip()


def message_role(message: dict[str, Any]) -> str:
    return normalize_role(message.get("role", message.get("from", "")))


def message_content(message: dict[str, Any]) -> str:
    return clean_content(message.get("content", message.get("value", message.get("text", ""))))


def segments_from_messages(messages: Any) -> list[TrainingSegment]:
    if not isinstance(messages, list):
        return []
    segments: list[TrainingSegment] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message_role(message)
        if role not in {"system", "user", "assistant"}:
            continue
        content = message_content(message)
        if content:
            segments.append(TrainingSegment(role, content, role == "assistant"))
    return compact_segments(segments)


def compact_segments(segments: list[TrainingSegment]) -> list[TrainingSegment]:
    out: list[TrainingSegment] = []
    for segment in segments:
        role = normalize_role(segment.role)
        if role not in {"system", "user", "assistant"}:
            continue
        content = clean_content(segment.content)
        if not content:
            continue
        normalized = TrainingSegment(role, content, role == "assistant")
        if out and out[-1].role == normalized.role:
            merged = out[-1].content + "\n" + normalized.content
            out[-1] = TrainingSegment(normalized.role, merged, normalized.train)
        else:
            out.append(normalized)
    return out


def trim_segments(segments: list[TrainingSegment], cfg: ChatMixConfig) -> list[TrainingSegment]:
    out: list[TrainingSegment] = []
    used_chars = 0
    turns = 0
    for segment in segments:
        is_turn = segment.role in {"user", "assistant"}
        if is_turn and turns >= cfg.max_turns:
            break
        next_chars = used_chars + len(segment.content)
        if out and next_chars > cfg.max_record_chars:
            break
        if next_chars > cfg.max_record_chars:
            continue
        out.append(segment)
        used_chars = next_chars
        turns += int(is_turn)

    while out and out[-1].role != "assistant":
        out.pop()
    while out and out[0].role == "assistant":
        out.pop(0)
    return out


def rendered_text(segments: list[TrainingSegment]) -> str:
    return "\n".join(f"{segment.role}: {segment.content}" for segment in segments)


def is_probably_english(text: str, min_ratio: float) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 20:
        return False
    ascii_letters = [ch for ch in letters if "a" <= ch.lower() <= "z"]
    return len(ascii_letters) / max(len(letters), 1) >= min_ratio


def has_bad_template(segments: list[TrainingSegment]) -> bool:
    assistant_text = "\n".join(segment.content for segment in segments if segment.role == "assistant").lower()
    return any(phrase in assistant_text for phrase in BAD_ASSISTANT_PHRASES)


def toxicity_value(record: dict[str, Any]) -> float | None:
    detoxify = record.get("detoxify")
    if not isinstance(detoxify, dict):
        return None
    value = detoxify.get("toxicity")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def filter_segments(segments: list[TrainingSegment], cfg: ChatMixConfig) -> list[TrainingSegment]:
    segments = trim_segments(compact_segments(segments), cfg)
    roles = [segment.role for segment in segments if segment.role in {"user", "assistant"}]
    if len(roles) < cfg.min_turns or "user" not in roles or "assistant" not in roles:
        return []
    if len("".join(segment.content for segment in segments)) < cfg.min_record_chars:
        return []
    if any(len(segment.content) > cfg.max_user_chars for segment in segments if segment.role == "user"):
        return []
    assistant_chars = sum(len(segment.content) for segment in segments if segment.role == "assistant")
    if assistant_chars < cfg.min_assistant_chars:
        return []
    if any(len(segment.content) > cfg.max_assistant_chars for segment in segments if segment.role == "assistant"):
        return []
    text = rendered_text(segments)
    if not is_probably_english(text, cfg.english_ratio):
        return []
    if has_bad_template(segments):
        return []
    return segments


def fingerprint_segments(segments: list[TrainingSegment]) -> str:
    normalized = "\n".join(
        f"{segment.role}:{' '.join(segment.content.lower().split())}"
        for segment in segments
        if segment.role in {"user", "assistant"}
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_hf_dataset(dataset: str, split: str, cfg: ChatMixConfig, *, streaming: bool | None = None):
    load_dataset = require_load_dataset()
    use_streaming = cfg.streaming if streaming is None else streaming
    ds = load_dataset(dataset, split=split, streaming=use_streaming)
    if use_streaming and cfg.shuffle_buffer > 0:
        ds = ds.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)
    elif not use_streaming:
        ds = ds.shuffle(seed=cfg.seed)
    return ds


def iter_ultrachat(cfg: ChatMixConfig) -> Iterable[tuple[list[TrainingSegment], dict[str, Any]]]:
    for index, record in enumerate(load_hf_dataset(cfg.ultrachat_dataset, cfg.ultrachat_split, cfg)):
        yield segments_from_messages(record.get("messages")), {
            "dataset": cfg.ultrachat_dataset,
            "split": cfg.ultrachat_split,
            "source_index": index,
            "prompt_id": record.get("prompt_id"),
        }


def iter_empathetic(cfg: ChatMixConfig) -> Iterable[tuple[list[TrainingSegment], dict[str, Any]]]:
    for index, record in enumerate(load_hf_dataset(cfg.empathetic_dataset, cfg.empathetic_split, cfg)):
        messages = record.get("conversations", record.get("messages", record.get("dialog", [])))
        yield segments_from_messages(messages), {
            "dataset": cfg.empathetic_dataset,
            "split": cfg.empathetic_split,
            "source_index": index,
            "conv_id": record.get("conv_id"),
            "emotion": record.get("emotion"),
        }


def iter_daily_dialog(cfg: ChatMixConfig) -> Iterable[tuple[list[TrainingSegment], dict[str, Any]]]:
    if cfg.daily_dialog_local.exists():
        records: list[tuple[list[TrainingSegment], dict[str, Any]]] = []
        for line_no, line in enumerate(cfg.daily_dialog_local.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = record_from_payload(json.loads(line), str(cfg.daily_dialog_local), line_no)
            records.append(
                (
                    list(record.segments),
                    {
                        "dataset": str(cfg.daily_dialog_local),
                        "source_index": line_no,
                        "source_id": record.record_id,
                    },
                )
            )
        rng = random.Random(cfg.seed)
        rng.shuffle(records)
        yield from records
        return

    for index, record in enumerate(load_hf_dataset(cfg.daily_dialog_dataset, cfg.daily_dialog_split, cfg)):
        turns = record.get("dialog", [])
        segments = [
            TrainingSegment("user" if i % 2 == 0 else "assistant", clean_content(turn), i % 2 == 1)
            for i, turn in enumerate(turns)
            if clean_content(turn)
        ]
        yield segments, {
            "dataset": cfg.daily_dialog_dataset,
            "split": cfg.daily_dialog_split,
            "source_index": index,
        }


def load_oasst_messages(cfg: ChatMixConfig) -> list[dict[str, Any]]:
    load_dataset = require_load_dataset()
    rows: list[dict[str, Any]] = []
    for split in [part.strip() for part in cfg.oasst_splits.split(",") if part.strip()]:
        ds = load_dataset(cfg.oasst_dataset, split=split, streaming=False)
        rows.extend(dict(record) for record in ds)
    return rows


def is_kept_oasst_record(record: dict[str, Any], cfg: ChatMixConfig) -> bool:
    if record.get("lang") != "en":
        return False
    if bool(record.get("deleted", False)):
        return False
    if record.get("review_result") is False:
        return False
    if toxicity_value(record) is not None and toxicity_value(record) > cfg.max_toxicity:
        return False
    if normalize_role(record.get("role")) not in {"user", "assistant"}:
        return False
    return bool(clean_content(record.get("text", "")))


def oasst_rank(record: dict[str, Any]) -> int:
    rank = record.get("rank")
    return 1000 if rank is None else int(rank)


def oasst_path_score(path: list[dict[str, Any]]) -> tuple[int, int]:
    return (sum(oasst_rank(record) for record in path if record.get("role") == "assistant"), -len(path))


def iter_oasst(cfg: ChatMixConfig) -> Iterable[tuple[list[TrainingSegment], dict[str, Any]]]:
    rows = [record for record in load_oasst_messages(cfg) if is_kept_oasst_record(record, cfg)]
    by_id = {record["message_id"]: record for record in rows}
    children: dict[str | None, list[str]] = {}
    for record in rows:
        parent_id = record.get("parent_id")
        if parent_id and parent_id not in by_id:
            parent_id = None
        children.setdefault(parent_id, []).append(record["message_id"])

    for sibling_ids in children.values():
        sibling_ids.sort(key=lambda message_id: oasst_rank(by_id[message_id]))

    paths: list[list[dict[str, Any]]] = []
    leaf_ids = [message_id for message_id in by_id if message_id not in children]
    for leaf_id in leaf_ids:
        path: list[dict[str, Any]] = []
        seen: set[str] = set()
        current_id: str | None = leaf_id
        while current_id and current_id in by_id and current_id not in seen:
            seen.add(current_id)
            record = by_id[current_id]
            path.append(record)
            current_id = record.get("parent_id")
        path.reverse()
        if path and normalize_role(path[-1].get("role")) == "assistant":
            paths.append(path)

    paths.sort(key=oasst_path_score)
    for index, path in enumerate(paths):
        segments = [
            TrainingSegment(normalize_role(record.get("role")), clean_content(record.get("text")), normalize_role(record.get("role")) == "assistant")
            for record in path
        ]
        tree_id = path[0].get("message_tree_id") if path else None
        yield compact_segments(segments), {
            "dataset": cfg.oasst_dataset,
            "splits": cfg.oasst_splits,
            "source_index": index,
            "message_tree_id": tree_id,
        }


def write_source(
    f,
    *,
    source_name: str,
    target_records: int,
    examples: Iterable[tuple[list[TrainingSegment], dict[str, Any]]],
    cfg: ChatMixConfig,
    seen: set[str],
    next_id: int,
    stats: dict[str, Any],
) -> int:
    if target_records <= 0:
        return next_id

    source_stats = stats["sources"].setdefault(
        source_name,
        {"target_records": target_records, "scanned": 0, "written": 0, "skipped": 0, "duplicates": 0},
    )
    start = time.time()
    for raw_segments, meta in examples:
        if source_stats["written"] >= target_records:
            break
        source_stats["scanned"] += 1
        segments = filter_segments(raw_segments, cfg)
        if not segments:
            source_stats["skipped"] += 1
            continue
        fingerprint = fingerprint_segments(segments)
        if fingerprint in seen:
            source_stats["duplicates"] += 1
            continue
        seen.add(fingerprint)

        payload = make_record(
            f"chatmix-{next_id:06d}",
            source_name,
            segments,
            {
                **meta,
                "mix_source": source_name,
            },
            task="sft",
        )
        raw = json.dumps(payload, ensure_ascii=False)
        f.write(raw + "\n")
        next_id += 1
        source_stats["written"] += 1
        stats["written_records"] += 1
        stats["written_chars"] += len(raw)
        if cfg.log_every > 0 and stats["written_records"] % cfg.log_every == 0:
            elapsed = time.time() - start
            print(
                f"written={stats['written_records']} source={source_name} "
                f"source_written={source_stats['written']} elapsed={elapsed:.1f}s"
            )
    return next_id


def write_chat_mix(cfg: ChatMixConfig = CHAT_MIX_CFG) -> dict[str, Any]:
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cfg.out_path.with_name(cfg.out_path.name + ".tmp")
    meta_path = cfg.out_path.with_name(cfg.out_path.name + ".meta.json")

    stats: dict[str, Any] = {
        "schema": "tl-corpus-v1",
        "written_records": 0,
        "written_chars": 0,
        "sources": {},
    }
    seen: set[str] = set()
    next_id = 1
    start = time.time()

    with tmp_path.open("w", encoding="utf-8") as f:
        for source_name, target_records, examples in (
            ("ultrachat_200k", cfg.ultrachat_records, iter_ultrachat(cfg)),
            ("oasst1", cfg.oasst_records, iter_oasst(cfg)),
            ("empathetic_dialogues", cfg.empathetic_records, iter_empathetic(cfg)),
            ("daily_dialog", cfg.daily_dialog_records, iter_daily_dialog(cfg)),
        ):
            next_id = write_source(
                f,
                source_name=source_name,
                target_records=target_records,
                examples=examples,
                cfg=cfg,
                seen=seen,
                next_id=next_id,
                stats=stats,
            )

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
    parser = argparse.ArgumentParser(description="Prepare a mixed short-chat tl-corpus-v1 JSONL corpus.")
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--shuffle-buffer", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--ultrachat-records", type=int, default=None)
    parser.add_argument("--oasst-records", type=int, default=None)
    parser.add_argument("--empathetic-records", type=int, default=None)
    parser.add_argument("--daily-dialog-records", type=int, default=None)
    parser.add_argument("--min-turns", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--min-record-chars", type=int, default=None)
    parser.add_argument("--max-record-chars", type=int, default=None)
    parser.add_argument("--max-user-chars", type=int, default=None)
    parser.add_argument("--min-assistant-chars", type=int, default=None)
    parser.add_argument("--max-assistant-chars", type=int, default=None)
    return parser.parse_args()


def cfg_from_args(args: argparse.Namespace) -> ChatMixConfig:
    updates: dict[str, Any] = {}
    for arg_name, field_name in (
        ("out_path", "out_path"),
        ("seed", "seed"),
        ("shuffle_buffer", "shuffle_buffer"),
        ("log_every", "log_every"),
        ("ultrachat_records", "ultrachat_records"),
        ("oasst_records", "oasst_records"),
        ("empathetic_records", "empathetic_records"),
        ("daily_dialog_records", "daily_dialog_records"),
        ("min_turns", "min_turns"),
        ("max_turns", "max_turns"),
        ("min_record_chars", "min_record_chars"),
        ("max_record_chars", "max_record_chars"),
        ("max_user_chars", "max_user_chars"),
        ("min_assistant_chars", "min_assistant_chars"),
        ("max_assistant_chars", "max_assistant_chars"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value
    if args.no_streaming:
        updates["streaming"] = False
    return ChatMixConfig(**{**asdict(CHAT_MIX_CFG), **updates})


def main() -> None:
    write_chat_mix(cfg_from_args(parse_args()))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
