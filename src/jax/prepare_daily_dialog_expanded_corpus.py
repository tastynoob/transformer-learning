"""Expand DailyDialog into many short, deduplicated SFT records.

DailyDialog only has about 11k train conversations, but each conversation has
several assistant turns. This script turns every assistant turn into one or
more short context windows, filters out long records, and never truncates text.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable
import unicodedata

try:
    from .corpus import TrainingSegment, coerce_text, make_record, normalize_role, render_segment
    from .tokenizer import HFWordPieceTokenizer
except ImportError:
    from corpus import TrainingSegment, coerce_text, make_record, normalize_role, render_segment
    from tokenizer import HFWordPieceTokenizer


@dataclass
class DailyDialogExpandedConfig:
    out_path: Path = Path("data/corpus/daily_dialog_expanded.tl.jsonl")
    tokenizer_json: Path = Path("runs/jax_text_lm_quality_sft_short/tokenizer.json")
    dataset: str = "OpenRL/daily_dialog"
    splits: str = "train,validation,test"
    seed: int = 0
    log_every: int = 5000

    # Total number of user/assistant turns kept in a record. Each assistant
    # target is emitted with each matching context size, then deduplicated.
    context_turns: str = "2,4,6,8,10,12"
    max_records: int | None = None

    min_record_tokens: int = 10
    max_record_tokens: int = 256
    min_response_tokens: int = 2
    max_response_tokens: int = 80
    max_prompt_tokens: int = 200
    max_record_chars: int = 1200
    max_response_chars: int = 400
    dedupe_prompts: bool = True


DAILY_DIALOG_EXPANDED_CFG = DailyDialogExpandedConfig()


def require_load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets. Install it before preparing DailyDialog.") from exc
    return load_dataset


def csv_parts(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def int_parts(value: str) -> list[int]:
    return sorted({int(part) for part in csv_parts(value) if int(part) > 1})


def jsonable_config(cfg: DailyDialogExpandedConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def clean_content(value: Any) -> str:
    text = unicodedata.normalize("NFC", coerce_text(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("_comma_", ",")
    return " ".join(text.split()).strip()


def make_segment(role: str, content: Any) -> TrainingSegment | None:
    role = normalize_role(role)
    if role not in {"user", "assistant"}:
        return None
    content = clean_content(content)
    if not content:
        return None
    return TrainingSegment(role, content, role == "assistant")


def compact_segments(segments: Iterable[TrainingSegment]) -> list[TrainingSegment]:
    out: list[TrainingSegment] = []
    for segment in segments:
        normalized = make_segment(segment.role, segment.content)
        if normalized is None:
            continue
        if out and out[-1].role == normalized.role:
            out[-1] = TrainingSegment(
                normalized.role,
                f"{out[-1].content}\n{normalized.content}",
                normalized.train,
            )
        else:
            out.append(normalized)
    return out


def alternating_segments(turns: Iterable[Any]) -> list[TrainingSegment]:
    segments: list[TrainingSegment] = []
    for index, turn in enumerate(turns):
        segment = make_segment("user" if index % 2 == 0 else "assistant", turn)
        if segment is not None:
            segments.append(segment)
    return compact_segments(segments)


def retarget_last_assistant(segments: list[TrainingSegment]) -> list[TrainingSegment]:
    last_assistant = max((i for i, segment in enumerate(segments) if segment.role == "assistant"), default=-1)
    return [
        TrainingSegment(segment.role, segment.content, segment.role == "assistant" and i == last_assistant)
        for i, segment in enumerate(segments)
    ]


def windows_from_dialog(segments: list[TrainingSegment], cfg: DailyDialogExpandedConfig) -> Iterable[list[TrainingSegment]]:
    segments = [segment for segment in compact_segments(segments) if segment.role in {"user", "assistant"}]
    for index, segment in enumerate(segments):
        if segment.role != "assistant":
            continue
        for turns in int_parts(cfg.context_turns):
            start = max(0, index - turns + 1)
            while start <= index and segments[start].role == "assistant":
                start += 1
            window = retarget_last_assistant(segments[start : index + 1])
            if window and window[0].role == "user" and window[-1].role == "assistant":
                yield window


def visible_prompt_key(segments: list[TrainingSegment]) -> str:
    return "\n".join(
        f"{segment.role}:{' '.join(segment.content.lower().split())}"
        for segment in segments
        if not segment.train
    )


def fingerprint_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_tokens(tokenizer: HFWordPieceTokenizer, text: str, *, add_bos: bool = False, add_eos: bool = False) -> int:
    return len(tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos, truncate=False, return_np=False))


def validate_window(
    segments: list[TrainingSegment],
    tokenizer: HFWordPieceTokenizer,
    cfg: DailyDialogExpandedConfig,
) -> tuple[bool, str, dict[str, int]]:
    if not segments or segments[-1].role != "assistant":
        return False, "bad_roles", {}
    record_text = "".join(render_segment(segment) for segment in segments).strip()
    prompt_text = "".join(render_segment(segment) for segment in segments if not segment.train).strip()
    response_text = "".join(render_segment(segment) for segment in segments if segment.train).strip()

    if len(record_text) > cfg.max_record_chars:
        return False, "too_long_record_chars", {"record_chars": len(record_text)}
    if len(response_text) > cfg.max_response_chars:
        return False, "too_long_response_chars", {"response_chars": len(response_text)}

    record_tokens = count_tokens(tokenizer, record_text, add_bos=True, add_eos=True)
    if record_tokens < cfg.min_record_tokens:
        return False, "too_short_record", {"record_tokens": record_tokens}
    if record_tokens > cfg.max_record_tokens:
        return False, "too_long_record", {"record_tokens": record_tokens}

    prompt_tokens = count_tokens(tokenizer, prompt_text)
    response_tokens = count_tokens(tokenizer, response_text)
    if prompt_tokens > cfg.max_prompt_tokens:
        return False, "too_long_prompt", {"prompt_tokens": prompt_tokens}
    if response_tokens < cfg.min_response_tokens:
        return False, "too_short_response", {"response_tokens": response_tokens}
    if response_tokens > cfg.max_response_tokens:
        return False, "too_long_response", {"response_tokens": response_tokens}

    return True, "ok", {
        "record_tokens": record_tokens,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
    }


def iter_daily_dialog(cfg: DailyDialogExpandedConfig):
    load_dataset = require_load_dataset()
    for split in csv_parts(cfg.splits):
        ds = load_dataset(cfg.dataset, split=split, streaming=False)
        for index, row in enumerate(ds):
            yield split, index, row


def write_daily_dialog_expanded(cfg: DailyDialogExpandedConfig = DAILY_DIALOG_EXPANDED_CFG) -> dict[str, Any]:
    if not cfg.tokenizer_json.exists():
        raise FileNotFoundError(f"tokenizer_json does not exist: {cfg.tokenizer_json}")
    tokenizer = HFWordPieceTokenizer.load(cfg.tokenizer_json)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cfg.out_path.with_name(cfg.out_path.name + ".tmp")
    meta_path = cfg.out_path.with_name(cfg.out_path.name + ".meta.json")

    stats: dict[str, Any] = {
        "schema": "tl-corpus-v1",
        "dataset": cfg.dataset,
        "splits": cfg.splits,
        "tokenizer_json": str(cfg.tokenizer_json),
        "scanned_dialogs": 0,
        "candidate_windows": 0,
        "written_records": 0,
        "written_chars": 0,
        "duplicates": 0,
        "prompt_duplicates": 0,
        "skip_reasons": {},
        "token_sums": {"record": 0, "prompt": 0, "response": 0},
    }
    seen_records: set[str] = set()
    seen_prompts: set[str] = set()
    start = time.time()

    with tmp_path.open("w", encoding="utf-8") as f:
        for split, source_index, row in iter_daily_dialog(cfg):
            stats["scanned_dialogs"] += 1
            dialog = row.get("dialog", [])
            segments = alternating_segments(dialog if isinstance(dialog, list) else [])
            for window in windows_from_dialog(segments, cfg):
                if cfg.max_records is not None and stats["written_records"] >= cfg.max_records:
                    break
                stats["candidate_windows"] += 1
                prompt_hash = fingerprint_text(visible_prompt_key(window))
                if cfg.dedupe_prompts and prompt_hash in seen_prompts:
                    stats["prompt_duplicates"] += 1
                    continue

                record_hash = fingerprint_text(
                    "\n".join(f"{segment.role}:{segment.content}:{int(segment.train)}" for segment in window)
                )
                if record_hash in seen_records:
                    stats["duplicates"] += 1
                    continue

                ok, reason, token_info = validate_window(window, tokenizer, cfg)
                if not ok:
                    stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
                    continue

                record_id = f"daily-expanded-{stats['written_records'] + 1:07d}"
                payload = make_record(
                    record_id,
                    "daily_dialog_expanded",
                    window,
                    {
                        "dataset": cfg.dataset,
                        "split": split,
                        "source_index": source_index,
                        "record_tokens": token_info["record_tokens"],
                        "prompt_tokens": token_info["prompt_tokens"],
                        "response_tokens": token_info["response_tokens"],
                    },
                    task="sft",
                )
                raw = json.dumps(payload, ensure_ascii=False)
                f.write(raw + "\n")
                seen_records.add(record_hash)
                seen_prompts.add(prompt_hash)
                stats["written_records"] += 1
                stats["written_chars"] += len(raw)
                stats["token_sums"]["record"] += token_info["record_tokens"]
                stats["token_sums"]["prompt"] += token_info["prompt_tokens"]
                stats["token_sums"]["response"] += token_info["response_tokens"]

                if cfg.log_every > 0 and stats["written_records"] % cfg.log_every == 0:
                    elapsed = time.time() - start
                    print(
                        f"written={stats['written_records']} scanned={stats['scanned_dialogs']} "
                        f"candidates={stats['candidate_windows']} elapsed={elapsed:.1f}s",
                        flush=True,
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
    parser = argparse.ArgumentParser(description="Expand DailyDialog into short deduplicated SFT records.")
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--tokenizer-json", type=Path, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--context-turns", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--min-record-tokens", type=int, default=None)
    parser.add_argument("--max-record-tokens", type=int, default=None)
    parser.add_argument("--min-response-tokens", type=int, default=None)
    parser.add_argument("--max-response-tokens", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--max-record-chars", type=int, default=None)
    parser.add_argument("--max-response-chars", type=int, default=None)
    parser.add_argument("--allow-duplicate-prompts", action="store_true")
    return parser.parse_args()


def cfg_from_args(args: argparse.Namespace) -> DailyDialogExpandedConfig:
    updates: dict[str, Any] = {}
    for arg_name, field_name in (
        ("out_path", "out_path"),
        ("tokenizer_json", "tokenizer_json"),
        ("dataset", "dataset"),
        ("splits", "splits"),
        ("seed", "seed"),
        ("log_every", "log_every"),
        ("context_turns", "context_turns"),
        ("max_records", "max_records"),
        ("min_record_tokens", "min_record_tokens"),
        ("max_record_tokens", "max_record_tokens"),
        ("min_response_tokens", "min_response_tokens"),
        ("max_response_tokens", "max_response_tokens"),
        ("max_prompt_tokens", "max_prompt_tokens"),
        ("max_record_chars", "max_record_chars"),
        ("max_response_chars", "max_response_chars"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value
    if args.allow_duplicate_prompts:
        updates["dedupe_prompts"] = False
    return DailyDialogExpandedConfig(**{**asdict(DAILY_DIALOG_EXPANDED_CFG), **updates})


def main() -> None:
    write_daily_dialog_expanded(cfg_from_args(parse_args()))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
