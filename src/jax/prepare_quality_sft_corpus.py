"""Build a high-signal short SFT corpus in tl-corpus-v1 format.

The script is intentionally strict: it never truncates records. A candidate is
rendered exactly as the trainer will see it, tokenized with the run tokenizer,
and discarded if it does not fit the configured token budget.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable
import unicodedata

try:
    from .corpus import TrainingSegment, coerce_text, make_record, normalize_role, render_record, render_segment
    from .tokenizer import HFWordPieceTokenizer
except ImportError:
    from corpus import TrainingSegment, coerce_text, make_record, normalize_role, render_record, render_segment
    from tokenizer import HFWordPieceTokenizer


BAD_PROMPT_PHRASES = (
    "generate a detailed and long answer",
    "write an article",
    "write a long",
    "produce a long",
    "long descriptive",
    "write a story",
    "write an essay",
)

GENERIC_PROMPTS = {
    "hi",
    "hello",
    "hey",
    "hi.",
    "hello.",
    "how are you",
    "how are you?",
    "how are you today",
    "how are you today?",
    "what are you doing",
    "what are you doing?",
    "tell me about yourself",
    "tell me more about yourself",
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    subset: str | None
    split: str
    fmt: str
    target_records: int


@dataclass
class QualitySFTConfig:
    out_path: Path = Path("data/corpus/quality_sft_short.tl.jsonl")
    tokenizer_json: Path = Path("runs/pretrain/tokenizer.json")
    seed: int = 0
    streaming: bool = True
    shuffle_buffer: int = 20000
    log_every: int = 5000
    max_total_records: int | None = 500000

    smol_smoltalk_records: int = 200000
    ultrafeedback_records: int = 100000
    smol_constraints_records: int = 60000
    smol_summarize_records: int = 60000
    smol_rewrite_records: int = 60000
    no_robots_records: int = 8000
    dolly_records: int = 10000
    slimorca_records: int = 100000
    openhermes_records: int = 100000
    ultrainteract_records: int = 50000
    tulu_records: int = 100000

    min_record_tokens: int = 16
    max_record_tokens: int = 512
    min_response_tokens: int = 4
    max_response_tokens: int = 220
    max_prompt_tokens: int = 384
    max_record_chars: int = 2600
    max_prompt_chars: int = 1900
    max_response_chars: int = 1200
    max_turns: int = 8
    min_english_ratio: float = 0.85
    dedupe_prompts: bool = True


QUALITY_SFT_CFG = QualitySFTConfig()


def require_load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets. Install it before preparing HF corpora.") from exc
    return load_dataset


def jsonable_config(cfg: QualitySFTConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def clean_content(value: Any) -> str:
    text = unicodedata.normalize("NFC", coerce_text(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    out: list[str] = []
    blank = 0
    for line in lines:
        if line.strip():
            blank = 0
            out.append(line)
        else:
            blank += 1
            if blank <= 1:
                out.append("")
    return "\n".join(out).strip()


def make_segment(role: str, content: Any, *, train: bool | None = None) -> TrainingSegment | None:
    role = normalize_role(role)
    if role not in {"system", "user", "assistant"}:
        return None
    content = clean_content(content)
    if not content:
        return None
    return TrainingSegment(role, content, role == "assistant" if train is None else train)


def compact_segments(segments: Iterable[TrainingSegment]) -> list[TrainingSegment]:
    out: list[TrainingSegment] = []
    for segment in segments:
        normalized = make_segment(segment.role, segment.content, train=segment.train)
        if normalized is None:
            continue
        if out and out[-1].role == normalized.role and normalized.role != "system":
            merged = out[-1].content + "\n" + normalized.content
            out[-1] = TrainingSegment(normalized.role, merged, out[-1].train or normalized.train)
        else:
            out.append(normalized)
    return out


def messages_to_segments(messages: Any) -> list[TrainingSegment]:
    if not isinstance(messages, list):
        return []
    segments: list[TrainingSegment] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_role(message.get("role", message.get("from", "")))
        content = message.get("content", message.get("value", ""))
        segment = make_segment(role, content)
        if segment is not None:
            segments.append(segment)
    return compact_segments(segments)


def conversations_to_segments(conversations: Any) -> list[TrainingSegment]:
    if not isinstance(conversations, list):
        return []
    segments: list[TrainingSegment] = []
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = normalize_role(message.get("from", message.get("role", "")))
        content = message.get("value", message.get("content", ""))
        segment = make_segment(role, content)
        if segment is not None:
            segments.append(segment)
    return compact_segments(segments)


def dolly_to_segments(record: dict[str, Any]) -> list[TrainingSegment]:
    instruction = clean_content(record.get("instruction", ""))
    context = clean_content(record.get("context", ""))
    response = clean_content(record.get("response", ""))
    if not instruction or not response:
        return []
    user = instruction if not context else f"{instruction}\n\nContext:\n{context}"
    return [
        TrainingSegment("user", user, False),
        TrainingSegment("assistant", response, True),
    ]


def instruction_response_to_segments(record: dict[str, Any]) -> list[TrainingSegment]:
    instruction = clean_content(record.get("instruction", record.get("prompt", "")))
    response = clean_content(record.get("response", record.get("output", "")))
    if not instruction or not response:
        return []
    return [
        TrainingSegment("user", instruction, False),
        TrainingSegment("assistant", response, True),
    ]


def source_specs(cfg: QualitySFTConfig) -> list[SourceSpec]:
    return [
        SourceSpec("smol_smoltalk", "HuggingFaceTB/smol-smoltalk", None, "train", "messages", cfg.smol_smoltalk_records),
        SourceSpec("ultrafeedback_sft", "HuggingFaceH4/ultrafeedback_binarized", None, "train_sft", "messages", cfg.ultrafeedback_records),
        SourceSpec("smoltalk_constraints", "HuggingFaceTB/smoltalk", "smol-constraints", "train", "messages", cfg.smol_constraints_records),
        SourceSpec("smoltalk_summarize", "HuggingFaceTB/smoltalk", "smol-summarize", "train", "messages", cfg.smol_summarize_records),
        SourceSpec("smoltalk_rewrite", "HuggingFaceTB/smoltalk", "smol-rewrite", "train", "messages", cfg.smol_rewrite_records),
        SourceSpec("no_robots", "HuggingFaceH4/no_robots", None, "train", "messages", cfg.no_robots_records),
        SourceSpec("dolly_15k", "databricks/databricks-dolly-15k", None, "train", "dolly", cfg.dolly_records),
        SourceSpec("slimorca_short", "Open-Orca/SlimOrca", None, "train", "conversations", cfg.slimorca_records),
        SourceSpec("openhermes_short", "teknium/OpenHermes-2.5", None, "train", "conversations", cfg.openhermes_records),
        SourceSpec("ultrainteract_short", "openbmb/UltraInteract_sft", None, "train", "instruction_response", cfg.ultrainteract_records),
        SourceSpec("tulu3_short", "allenai/tulu-3-sft-mixture", None, "train", "messages", cfg.tulu_records),
    ]


def iter_dataset(spec: SourceSpec, cfg: QualitySFTConfig) -> Iterable[dict[str, Any]]:
    load_dataset = require_load_dataset()
    args = [spec.dataset]
    if spec.subset is not None:
        args.append(spec.subset)
    ds = load_dataset(*args, split=spec.split, streaming=cfg.streaming)
    if cfg.streaming and cfg.shuffle_buffer > 0:
        ds = ds.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)
    elif not cfg.streaming:
        ds = ds.shuffle(seed=cfg.seed)
    return ds


def record_to_segments(record: dict[str, Any], spec: SourceSpec) -> list[TrainingSegment]:
    language = record.get("language")
    if isinstance(language, str) and language and language.lower() not in {"en", "english"}:
        return []
    if spec.fmt == "messages":
        return messages_to_segments(record.get("messages"))
    if spec.fmt == "conversations":
        return conversations_to_segments(record.get("conversations"))
    if spec.fmt == "dolly":
        return dolly_to_segments(record)
    if spec.fmt == "instruction_response":
        return instruction_response_to_segments(record)
    raise ValueError(f"unsupported source format: {spec.fmt}")


def retarget_last_assistant(segments: list[TrainingSegment]) -> list[TrainingSegment]:
    last_assistant = max((i for i, segment in enumerate(segments) if segment.role == "assistant"), default=-1)
    return [
        TrainingSegment(segment.role, segment.content, segment.role == "assistant" and i == last_assistant)
        for i, segment in enumerate(segments)
    ]


def prefix_windows(segments: list[TrainingSegment], cfg: QualitySFTConfig) -> Iterable[list[TrainingSegment]]:
    segments = compact_segments(segments)
    for index, segment in enumerate(segments):
        if segment.role != "assistant":
            continue
        window = segments[: index + 1]
        turns = [item for item in window if item.role in {"user", "assistant"}]
        if cfg.max_turns > 0 and len(turns) > cfg.max_turns:
            continue
        window = retarget_last_assistant(window)
        if any(item.role == "user" for item in window) and window[-1].role == "assistant":
            yield window


def visible_prompt_key(segments: list[TrainingSegment]) -> str:
    return "\n".join(
        f"{segment.role}:{' '.join(segment.content.lower().split())}"
        for segment in segments
        if not segment.train
    )


def fingerprint_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def last_user_text(segments: list[TrainingSegment]) -> str:
    users = [segment.content for segment in segments if segment.role == "user" and not segment.train]
    return users[-1] if users else ""


def is_probably_english(text: str, min_ratio: float) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 8:
        return True
    ascii_letters = [ch for ch in letters if "a" <= ch.lower() <= "z"]
    return len(ascii_letters) / max(len(letters), 1) >= min_ratio


def has_bad_prompt(segments: list[TrainingSegment]) -> bool:
    prompt = visible_prompt_key(segments)
    if any(phrase in prompt for phrase in BAD_PROMPT_PHRASES):
        return True
    last_user = " ".join(last_user_text(segments).lower().split())
    return last_user in GENERIC_PROMPTS


def count_tokens(tokenizer: HFWordPieceTokenizer, text: str, *, add_bos: bool = False, add_eos: bool = False) -> int:
    return len(tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos, truncate=False, return_np=False))


def validate_segments(
    segments: list[TrainingSegment],
    tokenizer: HFWordPieceTokenizer,
    cfg: QualitySFTConfig,
) -> tuple[bool, str, dict[str, int]]:
    if not segments or segments[-1].role != "assistant":
        return False, "bad_roles", {}
    if not any(segment.role == "user" for segment in segments):
        return False, "no_user", {}
    if has_bad_prompt(segments):
        return False, "bad_prompt", {}

    record_text = "".join(render_segment(segment) for segment in segments).strip()
    prompt_text = "".join(render_segment(segment) for segment in segments if not segment.train).strip()
    response_text = "".join(render_segment(segment) for segment in segments if segment.train).strip()

    if len(record_text) > cfg.max_record_chars:
        return False, "too_long_record_chars", {"record_chars": len(record_text)}
    if len(prompt_text) > cfg.max_prompt_chars:
        return False, "too_long_prompt_chars", {"prompt_chars": len(prompt_text)}
    if len(response_text) > cfg.max_response_chars:
        return False, "too_long_response_chars", {"response_chars": len(response_text)}

    if not is_probably_english(record_text, cfg.min_english_ratio):
        return False, "non_english", {}

    record_tokens = count_tokens(tokenizer, record_text, add_bos=True, add_eos=True)
    if record_tokens < cfg.min_record_tokens:
        return False, "too_short_record", {"record_tokens": record_tokens}
    if record_tokens > cfg.max_record_tokens:
        return False, "too_long_record", {"record_tokens": record_tokens}

    prompt_tokens = count_tokens(tokenizer, prompt_text)
    response_tokens = count_tokens(tokenizer, response_text)
    if prompt_tokens > cfg.max_prompt_tokens:
        return False, "too_long_prompt", {"record_tokens": record_tokens, "prompt_tokens": prompt_tokens}
    if response_tokens < cfg.min_response_tokens:
        return False, "too_short_response", {"record_tokens": record_tokens, "response_tokens": response_tokens}
    if response_tokens > cfg.max_response_tokens:
        return False, "too_long_response", {"record_tokens": record_tokens, "response_tokens": response_tokens}

    return True, "ok", {
        "record_tokens": record_tokens,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
    }


def write_source(
    f,
    *,
    spec: SourceSpec,
    cfg: QualitySFTConfig,
    tokenizer: HFWordPieceTokenizer,
    seen_prompts: set[str],
    next_id: int,
    stats: dict[str, Any],
) -> int:
    if spec.target_records <= 0:
        return next_id

    source_stats = stats["sources"].setdefault(
        spec.name,
        {
            "dataset": spec.dataset,
            "subset": spec.subset,
            "split": spec.split,
            "target_records": spec.target_records,
            "scanned_records": 0,
            "candidate_windows": 0,
            "written": 0,
            "skipped": 0,
            "prompt_duplicates": 0,
            "skip_reasons": {},
            "token_sums": {"record": 0, "prompt": 0, "response": 0},
        },
    )
    start = time.time()
    for raw_record in iter_dataset(spec, cfg):
        if source_stats["written"] >= spec.target_records:
            break
        if cfg.max_total_records is not None and stats["written_records"] >= cfg.max_total_records:
            break
        source_stats["scanned_records"] += 1
        segments = record_to_segments(raw_record, spec)
        for window in prefix_windows(segments, cfg):
            if source_stats["written"] >= spec.target_records:
                break
            if cfg.max_total_records is not None and stats["written_records"] >= cfg.max_total_records:
                break
            source_stats["candidate_windows"] += 1
            prompt_key = visible_prompt_key(window)
            prompt_hash = fingerprint_text(prompt_key)
            if cfg.dedupe_prompts and prompt_hash in seen_prompts:
                source_stats["prompt_duplicates"] += 1
                continue

            ok, reason, token_info = validate_segments(window, tokenizer, cfg)
            if not ok:
                source_stats["skipped"] += 1
                source_stats["skip_reasons"][reason] = source_stats["skip_reasons"].get(reason, 0) + 1
                continue

            payload = make_record(
                f"quality-sft-{next_id:07d}",
                spec.name,
                window,
                {
                    "dataset": spec.dataset,
                    "subset": spec.subset,
                    "split": spec.split,
                    "source_index": source_stats["scanned_records"],
                    "record_tokens": token_info["record_tokens"],
                    "prompt_tokens": token_info["prompt_tokens"],
                    "response_tokens": token_info["response_tokens"],
                },
                task="sft",
            )
            raw = json.dumps(payload, ensure_ascii=False)
            f.write(raw + "\n")
            seen_prompts.add(prompt_hash)
            next_id += 1
            source_stats["written"] += 1
            stats["written_records"] += 1
            stats["written_chars"] += len(raw)
            source_stats["token_sums"]["record"] += token_info["record_tokens"]
            source_stats["token_sums"]["prompt"] += token_info["prompt_tokens"]
            source_stats["token_sums"]["response"] += token_info["response_tokens"]

            if cfg.log_every > 0 and stats["written_records"] % cfg.log_every == 0:
                elapsed = time.time() - start
                print(
                    f"written={stats['written_records']} source={spec.name} "
                    f"source_written={source_stats['written']} scanned={source_stats['scanned_records']} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    return next_id


def write_quality_sft(cfg: QualitySFTConfig = QUALITY_SFT_CFG) -> dict[str, Any]:
    if not cfg.tokenizer_json.exists():
        raise FileNotFoundError(f"tokenizer_json does not exist: {cfg.tokenizer_json}")
    tokenizer = HFWordPieceTokenizer.load(cfg.tokenizer_json)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cfg.out_path.with_name(cfg.out_path.name + ".tmp")
    meta_path = cfg.out_path.with_name(cfg.out_path.name + ".meta.json")

    stats: dict[str, Any] = {
        "schema": "tl-corpus-v1",
        "tokenizer_json": str(cfg.tokenizer_json),
        "written_records": 0,
        "written_chars": 0,
        "sources": {},
    }
    seen_prompts: set[str] = set()
    next_id = 1
    start = time.time()

    with tmp_path.open("w", encoding="utf-8") as f:
        for spec in source_specs(cfg):
            if cfg.max_total_records is not None and stats["written_records"] >= cfg.max_total_records:
                break
            print(f"source={spec.name} target_records={spec.target_records}", flush=True)
            next_id = write_source(
                f,
                spec=spec,
                cfg=cfg,
                tokenizer=tokenizer,
                seen_prompts=seen_prompts,
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
    parser = argparse.ArgumentParser(description="Prepare a high-signal short SFT tl-corpus-v1 JSONL corpus.")
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--tokenizer-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--shuffle-buffer", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--max-total-records", type=int, default=None)
    parser.add_argument("--no-total-limit", action="store_true")
    parser.add_argument("--ultrafeedback-records", type=int, default=None)
    parser.add_argument("--smol-rewrite-records", type=int, default=None)
    parser.add_argument("--smol-summarize-records", type=int, default=None)
    parser.add_argument("--smol-constraints-records", type=int, default=None)
    parser.add_argument("--no-robots-records", type=int, default=None)
    parser.add_argument("--dolly-records", type=int, default=None)
    parser.add_argument("--slimorca-records", type=int, default=None)
    parser.add_argument("--smol-smoltalk-records", type=int, default=None)
    parser.add_argument("--openhermes-records", type=int, default=None)
    parser.add_argument("--ultrainteract-records", type=int, default=None)
    parser.add_argument("--tulu-records", type=int, default=None)
    parser.add_argument("--min-record-tokens", type=int, default=None)
    parser.add_argument("--max-record-tokens", type=int, default=None)
    parser.add_argument("--min-response-tokens", type=int, default=None)
    parser.add_argument("--max-response-tokens", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--max-record-chars", type=int, default=None)
    parser.add_argument("--max-prompt-chars", type=int, default=None)
    parser.add_argument("--max-response-chars", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--min-english-ratio", type=float, default=None)
    parser.add_argument("--allow-duplicate-prompts", action="store_true")
    return parser.parse_args()


def cfg_from_args(args: argparse.Namespace) -> QualitySFTConfig:
    updates: dict[str, Any] = {}
    for arg_name, field_name in (
        ("out_path", "out_path"),
        ("tokenizer_json", "tokenizer_json"),
        ("seed", "seed"),
        ("shuffle_buffer", "shuffle_buffer"),
        ("log_every", "log_every"),
        ("max_total_records", "max_total_records"),
        ("ultrafeedback_records", "ultrafeedback_records"),
        ("smol_rewrite_records", "smol_rewrite_records"),
        ("smol_summarize_records", "smol_summarize_records"),
        ("smol_constraints_records", "smol_constraints_records"),
        ("no_robots_records", "no_robots_records"),
        ("dolly_records", "dolly_records"),
        ("slimorca_records", "slimorca_records"),
        ("smol_smoltalk_records", "smol_smoltalk_records"),
        ("openhermes_records", "openhermes_records"),
        ("ultrainteract_records", "ultrainteract_records"),
        ("tulu_records", "tulu_records"),
        ("min_record_tokens", "min_record_tokens"),
        ("max_record_tokens", "max_record_tokens"),
        ("min_response_tokens", "min_response_tokens"),
        ("max_response_tokens", "max_response_tokens"),
        ("max_prompt_tokens", "max_prompt_tokens"),
        ("max_record_chars", "max_record_chars"),
        ("max_prompt_chars", "max_prompt_chars"),
        ("max_response_chars", "max_response_chars"),
        ("max_turns", "max_turns"),
        ("min_english_ratio", "min_english_ratio"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value
    if args.no_streaming:
        updates["streaming"] = False
    if args.no_total_limit:
        updates["max_total_records"] = None
    if args.allow_duplicate_prompts:
        updates["dedupe_prompts"] = False
    return QualitySFTConfig(**{**asdict(QUALITY_SFT_CFG), **updates})


def main() -> None:
    write_quality_sft(cfg_from_args(parse_args()))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
