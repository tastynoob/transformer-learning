"""Prepare a short Chinese dialogue corpus from LCCC.

The Hugging Face `thu-coai/lccc` dataset is an old dataset-script wrapper. New
versions of `datasets` no longer execute those scripts, so this script reads
the raw LCCC files used by that wrapper directly from `silver/lccc`.

The output is tl-corpus-v1 JSONL. Each record is a short context window ending
at one assistant turn. Long records are filtered instead of truncated.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
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
    from .corpus import TrainingSegment, coerce_text, make_record, normalize_role, render_segment
    from .tokenizer import HFWordPieceTokenizer
except ImportError:
    from corpus import TrainingSegment, coerce_text, make_record, normalize_role, render_segment
    from tokenizer import HFWordPieceTokenizer


URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")


@dataclass
class LCCCZhConfig:
    out_path: Path = Path("data/corpus/lccc_zh_short.tl.jsonl")
    tokenizer_json: Path = Path("runs/jax_text_lm_daily_dialog_expanded/tokenizer.json")
    dataset: str = "thu-coai/lccc"
    source_repo: str = "silver/lccc"
    source_files: str = "lccc_base_train.jsonl.gz,lccc_base_valid.jsonl.gz,lccc_base_test.jsonl.gz"
    seed: int = 0
    log_every: int = 10000

    max_records: int | None = 300000
    context_turns: str = "2,4"
    dedupe_prompts: bool = True

    min_record_tokens: int = 8
    max_record_tokens: int = 256
    min_response_tokens: int = 2
    max_response_tokens: int = 80
    max_prompt_tokens: int = 200
    min_record_chars: int = 4
    max_record_chars: int = 360
    min_response_chars: int = 2
    max_response_chars: int = 90
    min_cjk_chars_per_segment: int = 1
    min_cjk_ratio: float = 0.55
    max_ascii_letters: int = 0


LCCC_ZH_CFG = LCCCZhConfig()


def require_hf_download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Missing dependency: huggingface_hub. Install it before preparing LCCC.") from exc
    return hf_hub_download


def csv_parts(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def int_parts(value: str) -> list[int]:
    return sorted({int(part) for part in csv_parts(value) if int(part) > 1})


def jsonable_config(cfg: LCCCZhConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def is_cjk_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def is_punctuation(ch: str) -> bool:
    return bool(ch) and unicodedata.category(ch).startswith("P")


def is_tight_zh_char(ch: str) -> bool:
    return is_cjk_char(ch) or ch.isdigit() or is_punctuation(ch)


def cjk_count(text: str) -> int:
    return sum(1 for ch in text if is_cjk_char(ch))


def content_ratio_base(text: str) -> int:
    return sum(1 for ch in text if is_cjk_char(ch) or ch.isascii() and ch.isalnum())


def chinese_ratio(text: str) -> float:
    base = content_ratio_base(text)
    if base == 0:
        return 0.0
    return cjk_count(text) / base


def remove_cjk_spacing(text: str) -> str:
    chars: list[str] = []
    for index, ch in enumerate(text):
        if ch != " ":
            chars.append(ch)
            continue
        prev_ch = text[index - 1] if index > 0 else ""
        next_ch = text[index + 1] if index + 1 < len(text) else ""
        if (
            is_tight_zh_char(prev_ch)
            and is_tight_zh_char(next_ch)
            and (is_cjk_char(prev_ch) or is_cjk_char(next_ch) or is_punctuation(prev_ch) or is_punctuation(next_ch))
        ):
            continue
        chars.append(ch)
    return "".join(chars)


def clean_content(value: Any) -> str:
    text = unicodedata.normalize("NFC", coerce_text(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("_comma_", "，")
    text = " ".join(text.split())
    text = remove_cjk_spacing(text)
    return text.strip()


def bad_content(text: str, cfg: LCCCZhConfig) -> bool:
    if not text:
        return True
    if URL_RE.search(text) or "@" in text:
        return True
    if len(ASCII_ALPHA_RE.findall(text)) > cfg.max_ascii_letters:
        return True
    if cjk_count(text) < cfg.min_cjk_chars_per_segment:
        return True
    return chinese_ratio(text) < cfg.min_cjk_ratio


def make_segment(role: str, content: Any, cfg: LCCCZhConfig) -> TrainingSegment | None:
    role = normalize_role(role)
    if role not in {"user", "assistant"}:
        return None
    content = clean_content(content)
    if bad_content(content, cfg):
        return None
    return TrainingSegment(role, content, role == "assistant")


def compact_segments(segments: Iterable[TrainingSegment], cfg: LCCCZhConfig) -> list[TrainingSegment]:
    out: list[TrainingSegment] = []
    for segment in segments:
        normalized = make_segment(segment.role, segment.content, cfg)
        if normalized is None:
            continue
        if out and out[-1].role == normalized.role:
            merged = f"{out[-1].content}\n{normalized.content}"
            out[-1] = TrainingSegment(normalized.role, merged, normalized.train)
        else:
            out.append(normalized)
    return out


def alternating_segments(turns: Iterable[Any], cfg: LCCCZhConfig) -> list[TrainingSegment]:
    segments: list[TrainingSegment] = []
    for index, turn in enumerate(turns):
        role = "user" if index % 2 == 0 else "assistant"
        segment = make_segment(role, turn, cfg)
        if segment is not None:
            segments.append(segment)
    return compact_segments(segments, cfg)


def retarget_last_assistant(segments: list[TrainingSegment]) -> list[TrainingSegment]:
    last_assistant = max((i for i, segment in enumerate(segments) if segment.role == "assistant"), default=-1)
    return [
        TrainingSegment(segment.role, segment.content, segment.role == "assistant" and i == last_assistant)
        for i, segment in enumerate(segments)
    ]


def windows_from_dialog(segments: list[TrainingSegment], cfg: LCCCZhConfig) -> Iterable[list[TrainingSegment]]:
    compacted = [segment for segment in compact_segments(segments, cfg) if segment.role in {"user", "assistant"}]
    for index, segment in enumerate(compacted):
        if segment.role != "assistant":
            continue
        for turns in int_parts(cfg.context_turns):
            start = max(0, index - turns + 1)
            while start <= index and compacted[start].role == "assistant":
                start += 1
            window = retarget_last_assistant(compacted[start : index + 1])
            if window and window[0].role == "user" and window[-1].role == "assistant":
                yield window


def fingerprint_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def visible_prompt_key(segments: list[TrainingSegment]) -> str:
    return "\n".join(
        f"{segment.role}:{' '.join(segment.content.lower().split())}"
        for segment in segments
        if not segment.train
    )


def record_key(segments: list[TrainingSegment]) -> str:
    return "\n".join(
        f"{segment.role}:{' '.join(segment.content.lower().split())}:{int(segment.train)}"
        for segment in segments
    )


def count_tokens(tokenizer: HFWordPieceTokenizer, text: str, *, add_bos: bool = False, add_eos: bool = False) -> int:
    return len(tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos, truncate=False, return_np=False))


def validate_window(
    segments: list[TrainingSegment],
    tokenizer: HFWordPieceTokenizer,
    cfg: LCCCZhConfig,
) -> tuple[bool, str, dict[str, int]]:
    if not segments or segments[-1].role != "assistant":
        return False, "bad_roles", {}
    if any(bad_content(segment.content, cfg) for segment in segments):
        return False, "non_chinese", {}

    record_text = "".join(render_segment(segment) for segment in segments).strip()
    prompt_text = "".join(render_segment(segment) for segment in segments if not segment.train).strip()
    response_text = "".join(render_segment(segment) for segment in segments if segment.train).strip()

    if len(record_text) < cfg.min_record_chars:
        return False, "too_short_record_chars", {"record_chars": len(record_text)}
    if len(record_text) > cfg.max_record_chars:
        return False, "too_long_record_chars", {"record_chars": len(record_text)}
    if len(response_text) < cfg.min_response_chars:
        return False, "too_short_response_chars", {"response_chars": len(response_text)}
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


def hf_download(repo_id: str, filename: str) -> Path:
    hf_hub_download = require_hf_download()
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))


def iter_lccc_dialogs(cfg: LCCCZhConfig):
    for filename in csv_parts(cfg.source_files):
        path = hf_download(cfg.source_repo, filename)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    dialog = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(dialog, list):
                    yield filename, line_no, dialog


def write_lccc_zh_corpus(cfg: LCCCZhConfig = LCCC_ZH_CFG) -> dict[str, Any]:
    if not cfg.tokenizer_json.exists():
        raise FileNotFoundError(f"tokenizer_json does not exist: {cfg.tokenizer_json}")
    tokenizer = HFWordPieceTokenizer.load(cfg.tokenizer_json)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cfg.out_path.with_name(cfg.out_path.name + ".tmp")
    meta_path = cfg.out_path.with_name(cfg.out_path.name + ".meta.json")

    stats: dict[str, Any] = {
        "schema": "tl-corpus-v1",
        "dataset": cfg.dataset,
        "source_repo": cfg.source_repo,
        "source_files": cfg.source_files,
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
        for source_file, source_index, dialog in iter_lccc_dialogs(cfg):
            if cfg.max_records is not None and stats["written_records"] >= cfg.max_records:
                break
            stats["scanned_dialogs"] += 1
            segments = alternating_segments(dialog, cfg)
            for window in windows_from_dialog(segments, cfg):
                if cfg.max_records is not None and stats["written_records"] >= cfg.max_records:
                    break
                stats["candidate_windows"] += 1

                prompt_hash = fingerprint_text(visible_prompt_key(window))
                if cfg.dedupe_prompts and prompt_hash in seen_prompts:
                    stats["prompt_duplicates"] += 1
                    continue

                rec_hash = fingerprint_text(record_key(window))
                if rec_hash in seen_records:
                    stats["duplicates"] += 1
                    continue

                ok, reason, token_info = validate_window(window, tokenizer, cfg)
                if not ok:
                    stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
                    continue

                record_id = f"lccc-zh-{stats['written_records'] + 1:08d}"
                payload = make_record(
                    record_id,
                    "lccc_zh_short",
                    window,
                    {
                        "dataset": cfg.dataset,
                        "source_repo": cfg.source_repo,
                        "source_file": source_file,
                        "source_index": source_index,
                        "record_tokens": token_info["record_tokens"],
                        "prompt_tokens": token_info["prompt_tokens"],
                        "response_tokens": token_info["response_tokens"],
                    },
                    task="sft",
                )
                raw = json.dumps(payload, ensure_ascii=False)
                f.write(raw + "\n")
                seen_records.add(rec_hash)
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
    parser = argparse.ArgumentParser(description="Prepare short Chinese LCCC dialogue corpus.")
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--tokenizer-json", type=Path, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--source-repo", default=None)
    parser.add_argument("--source-files", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--context-turns", default=None)
    parser.add_argument("--min-record-tokens", type=int, default=None)
    parser.add_argument("--max-record-tokens", type=int, default=None)
    parser.add_argument("--min-response-tokens", type=int, default=None)
    parser.add_argument("--max-response-tokens", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--min-record-chars", type=int, default=None)
    parser.add_argument("--max-record-chars", type=int, default=None)
    parser.add_argument("--min-response-chars", type=int, default=None)
    parser.add_argument("--max-response-chars", type=int, default=None)
    parser.add_argument("--min-cjk-chars-per-segment", type=int, default=None)
    parser.add_argument("--min-cjk-ratio", type=float, default=None)
    parser.add_argument("--max-ascii-letters", type=int, default=None)
    parser.add_argument("--allow-duplicate-prompts", action="store_true")
    return parser.parse_args()


def cfg_from_args(args: argparse.Namespace) -> LCCCZhConfig:
    updates: dict[str, Any] = {}
    for arg_name, field_name in (
        ("out_path", "out_path"),
        ("tokenizer_json", "tokenizer_json"),
        ("dataset", "dataset"),
        ("source_repo", "source_repo"),
        ("source_files", "source_files"),
        ("seed", "seed"),
        ("log_every", "log_every"),
        ("max_records", "max_records"),
        ("context_turns", "context_turns"),
        ("min_record_tokens", "min_record_tokens"),
        ("max_record_tokens", "max_record_tokens"),
        ("min_response_tokens", "min_response_tokens"),
        ("max_response_tokens", "max_response_tokens"),
        ("max_prompt_tokens", "max_prompt_tokens"),
        ("min_record_chars", "min_record_chars"),
        ("max_record_chars", "max_record_chars"),
        ("min_response_chars", "min_response_chars"),
        ("max_response_chars", "max_response_chars"),
        ("min_cjk_chars_per_segment", "min_cjk_chars_per_segment"),
        ("min_cjk_ratio", "min_cjk_ratio"),
        ("max_ascii_letters", "max_ascii_letters"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value
    if args.allow_duplicate_prompts:
        updates["dedupe_prompts"] = False
    return LCCCZhConfig(**{**asdict(LCCC_ZH_CFG), **updates})


def main() -> None:
    write_lccc_zh_corpus(cfg_from_args(parse_args()))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
