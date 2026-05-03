"""Prepare a Chinese logic-heavy dialogue corpus.

This script mixes DuConv, CrossWOZ, KdConv, and RiSAWOZ into tl-corpus-v1
records. The default mode keeps full dialogue prefixes up to each assistant
turn, so each sample has complete visible context instead of arbitrary middle
windows.
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
from urllib.request import urlretrieve
import zipfile

try:
    from .corpus import TrainingSegment, coerce_text, make_record, normalize_role, render_segment
    from .tokenizer import HFWordPieceTokenizer
except ImportError:
    from corpus import TrainingSegment, coerce_text, make_record, normalize_role, render_segment
    from tokenizer import HFWordPieceTokenizer


DUCONV_URL = "https://bj.bcebos.com/paddlenlp/datasets/DuConv.zip"


@dataclass
class LogicDialogConfig:
    out_path: Path = Path("data/corpus/zh_logic_dialog_mix.tl.jsonl")
    tokenizer_json: Path = Path("runs/jax_text_lm_quality_sft_short/tokenizer.json")
    seed: int = 0
    log_every: int = 10000

    duconv_records: int = 120000
    crosswoz_records: int = 120000
    kdconv_records: int = 120000
    risawoz_records: int = 120000

    # full: one complete dialogue per record
    # prefix: full dialogue prefix up to each assistant turn
    # single_turn: only adjacent user->assistant pairs
    # window: legacy overlapping context windows
    record_mode: str = "prefix"
    context_turns: str = "2,4,6,8"
    dedupe_prompts: bool = True

    min_record_tokens: int = 8
    max_record_tokens: int = 512
    min_response_tokens: int = 2
    max_response_tokens: int = 96
    max_prompt_tokens: int = 480
    min_record_chars: int = 4
    max_record_chars: int = 1600
    min_response_chars: int = 2
    max_response_chars: int = 160
    min_cjk_ratio: float = 0.55
    max_ascii_letters: int = 0
    source_dir: Path = Path("data/raw/zh_logic_dialog")


LOGIC_DIALOG_CFG = LogicDialogConfig()


def require_hf_download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Missing dependency: huggingface_hub. Install it before preparing the corpus.") from exc
    return hf_hub_download


def csv_parts(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def int_parts(value: str) -> list[int]:
    return sorted({int(part) for part in csv_parts(value) if int(part) > 1})


def jsonable_config(cfg: LogicDialogConfig) -> dict[str, Any]:
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
    return sum(1 for ch in text if is_cjk_char(ch) or (ch.isascii() and ch.isalnum()))


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


def normalize_dialog_role(role: Any) -> str:
    value = normalize_role(role)
    if value == "usr":
        return "user"
    if value == "sys":
        return "assistant"
    return value


def bad_content(text: str, cfg: LogicDialogConfig) -> bool:
    if not text:
        return True
    if len(re.findall(r"[A-Za-z]", text)) > cfg.max_ascii_letters:
        return True
    return chinese_ratio(text) < cfg.min_cjk_ratio


def make_segment(role: str, content: Any, cfg: LogicDialogConfig) -> TrainingSegment | None:
    role = normalize_dialog_role(role)
    if role not in {"system", "user", "assistant"}:
        return None
    content = clean_content(content)
    if bad_content(content, cfg):
        return None
    return TrainingSegment(role, content, role == "assistant")


def compact_segments(segments: Iterable[TrainingSegment], cfg: LogicDialogConfig) -> list[TrainingSegment]:
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


def split_system_prefix(segments: list[TrainingSegment]) -> tuple[list[TrainingSegment], list[TrainingSegment]]:
    system_prefix: list[TrainingSegment] = []
    index = 0
    while index < len(segments) and segments[index].role == "system":
        system_prefix.append(segments[index])
        index += 1
    dialogue = [segment for segment in segments[index:] if segment.role in {"user", "assistant"}]
    while dialogue and dialogue[0].role != "user":
        dialogue = dialogue[1:]
    while dialogue and dialogue[-1].role != "assistant":
        dialogue = dialogue[:-1]
    return system_prefix, dialogue


def alternating_segments(turns: Iterable[Any], cfg: LogicDialogConfig) -> list[TrainingSegment]:
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


def full_dialog_from_dialog(segments: list[TrainingSegment], cfg: LogicDialogConfig) -> list[TrainingSegment]:
    compacted = compact_segments(segments, cfg)
    system_prefix, dialogue = split_system_prefix(compacted)
    return system_prefix + [
        TrainingSegment(segment.role, segment.content, segment.role == "assistant")
        for segment in dialogue
    ]


def prefix_records_from_dialog(segments: list[TrainingSegment], cfg: LogicDialogConfig) -> Iterable[list[TrainingSegment]]:
    compacted = compact_segments(segments, cfg)
    system_prefix, dialogue = split_system_prefix(compacted)
    for index, segment in enumerate(dialogue):
        if segment.role != "assistant":
            continue
        prefix = system_prefix + retarget_last_assistant(dialogue[: index + 1])
        if prefix:
            yield prefix


def single_turn_records_from_dialog(segments: list[TrainingSegment], cfg: LogicDialogConfig) -> Iterable[list[TrainingSegment]]:
    compacted = compact_segments(segments, cfg)
    system_prefix, dialogue = split_system_prefix(compacted)
    for index in range(1, len(dialogue)):
        if dialogue[index - 1].role != "user" or dialogue[index].role != "assistant":
            continue
        window = system_prefix + retarget_last_assistant(dialogue[index - 1 : index + 1])
        if window:
            yield window


def window_records_from_dialog(segments: list[TrainingSegment], cfg: LogicDialogConfig) -> Iterable[list[TrainingSegment]]:
    compacted = compact_segments(segments, cfg)
    system_prefix, dialogue = split_system_prefix(compacted)
    for index, segment in enumerate(dialogue):
        if segment.role != "assistant":
            continue
        for turns in int_parts(cfg.context_turns):
            start = max(0, index - turns + 1)
            while start <= index and dialogue[start].role == "assistant":
                start += 1
            window = system_prefix + retarget_last_assistant(dialogue[start : index + 1])
            if window and window[-1].role == "assistant":
                yield window


def records_from_dialog(segments: list[TrainingSegment], cfg: LogicDialogConfig) -> Iterable[list[TrainingSegment]]:
    mode = cfg.record_mode.strip().lower()
    if mode == "full":
        full = full_dialog_from_dialog(segments, cfg)
        if full:
            yield full
        return
    if mode == "prefix":
        yield from prefix_records_from_dialog(segments, cfg)
        return
    if mode == "single_turn":
        yield from single_turn_records_from_dialog(segments, cfg)
        return
    if mode == "window":
        yield from window_records_from_dialog(segments, cfg)
        return
    raise ValueError("record_mode must be one of: full, prefix, single_turn, window")


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


def validate_record(
    segments: list[TrainingSegment],
    tokenizer: HFWordPieceTokenizer,
    cfg: LogicDialogConfig,
) -> tuple[bool, str, dict[str, int]]:
    if not segments or segments[-1].role != "assistant":
        return False, "bad_roles", {}

    record_text = "".join(render_segment(segment) for segment in segments).strip()
    prompt_text = "".join(render_segment(segment) for segment in segments if not segment.train).strip()

    if any(bad_content(segment.content, cfg) for segment in segments):
        return False, "non_chinese", {}
    if len(record_text) < cfg.min_record_chars:
        return False, "too_short_record_chars", {"record_chars": len(record_text)}
    if len(record_text) > cfg.max_record_chars:
        return False, "too_long_record_chars", {"record_chars": len(record_text)}

    record_tokens = count_tokens(tokenizer, record_text, add_bos=True, add_eos=True)
    if record_tokens < cfg.min_record_tokens:
        return False, "too_short_record", {"record_tokens": record_tokens}
    if record_tokens > cfg.max_record_tokens:
        return False, "too_long_record", {"record_tokens": record_tokens}

    prompt_tokens = count_tokens(tokenizer, prompt_text)
    if prompt_tokens > cfg.max_prompt_tokens:
        return False, "too_long_prompt", {"prompt_tokens": prompt_tokens}

    response_tokens = 0
    response_chars = 0
    for segment in segments:
        if not segment.train:
            continue
        response_chars += len(segment.content)
        segment_text = render_segment(segment).strip()
        segment_tokens = count_tokens(tokenizer, segment_text, add_bos=False, add_eos=False)
        if len(segment.content) < cfg.min_response_chars:
            return False, "too_short_response_chars", {"response_chars": len(segment.content)}
        if len(segment.content) > cfg.max_response_chars:
            return False, "too_long_response_chars", {"response_chars": len(segment.content)}
        if segment_tokens < cfg.min_response_tokens:
            return False, "too_short_response", {"response_tokens": segment_tokens}
        if segment_tokens > cfg.max_response_tokens:
            return False, "too_long_response", {"response_tokens": segment_tokens}
        response_tokens += segment_tokens

    return True, "ok", {
        "record_tokens": record_tokens,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "response_chars": response_chars,
    }


def download_external(url: str, raw_dir: Path, filename: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    print(f"downloading {url} -> {path}")
    urlretrieve(url, tmp_path)
    tmp_path.replace(path)
    return path


def hf_download(repo_id: str, filename: str) -> Path:
    hf_hub_download = require_hf_download()
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))


def iter_duconv(cfg: LogicDialogConfig) -> Iterable[tuple[list[TrainingSegment], dict[str, Any]]]:
    archive_path = download_external(DUCONV_URL, cfg.source_dir, "DuConv.zip")
    with zipfile.ZipFile(archive_path) as archive:
        for filename in ("DuConv/train.txt", "DuConv/dev.txt", "DuConv/test_1.txt", "DuConv/test_2.txt"):
            with archive.open(filename) as f:
                for line_no, raw_line in enumerate(f, start=1):
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    segments: list[TrainingSegment] = []
                    if isinstance(record.get("conversation"), list) and record["conversation"]:
                        segments = alternating_segments(record["conversation"], cfg)
                    else:
                        history = record.get("history")
                        if isinstance(history, list) and history:
                            segments = alternating_segments(history, cfg)
                            response = make_segment("assistant", record.get("response", ""), cfg)
                            if response is not None:
                                segments.append(response)
                                segments = compact_segments(segments, cfg)

                    if segments:
                        yield segments, {
                            "dataset": "PaddlePaddle/duconv",
                            "source_file": filename,
                            "source_index": line_no,
                            "goal_len": len(record.get("goal", [])) if isinstance(record.get("goal", []), list) else 0,
                            "knowledge_len": len(record.get("knowledge", [])) if isinstance(record.get("knowledge", []), list) else 0,
                        }


def iter_kdconv(cfg: LogicDialogConfig) -> Iterable[tuple[list[TrainingSegment], dict[str, Any]]]:
    local_archive = cfg.source_dir / "kd_conv_with_kb" / "data.zip"
    archive_path = local_archive if local_archive.exists() else hf_download("thu-coai/kd_conv_with_kb", "data.zip")
    with zipfile.ZipFile(archive_path) as archive:
        for domain in ("film", "music", "travel"):
            for split in ("train", "dev", "test"):
                filename = f"data/{domain}/{split}.json"
                with archive.open(filename) as f:
                    data = json.loads(f.read().decode("utf-8"))
                if not isinstance(data, list):
                    continue
                for dialog_index, dialog in enumerate(data):
                    messages = dialog.get("messages", [])
                    if not isinstance(messages, list) or not messages:
                        continue
                    segments: list[TrainingSegment] = []
                    name = clean_content(dialog.get("name", ""))
                    if name:
                        system = make_segment("system", f"话题：{name}", cfg)
                        if system is not None:
                            segments.append(system)
                    for index, message in enumerate(messages):
                        if not isinstance(message, dict):
                            continue
                        role = "user" if index % 2 == 0 else "assistant"
                        segment = make_segment(role, message.get("message", ""), cfg)
                        if segment is not None:
                            segments.append(segment)
                    if segments:
                        yield segments, {
                            "dataset": "thu-coai/kd_conv_with_kb",
                            "domain": domain,
                            "split": split,
                            "dialog_index": dialog_index,
                            "name": name,
                            "message_count": len(messages),
                        }


def iter_crosswoz(cfg: LogicDialogConfig) -> Iterable[tuple[list[TrainingSegment], dict[str, Any]]]:
    archive_path = hf_download("GEM/CrossWOZ", "data.zip")
    with zipfile.ZipFile(archive_path) as archive:
        for filename in ("train.json", "val.json", "test.json"):
            with archive.open(filename) as f:
                data = json.loads(f.read().decode("utf-8"))
            if not isinstance(data, dict):
                continue
            for dialog_id, dialog in data.items():
                messages = dialog.get("messages", [])
                if not isinstance(messages, list):
                    continue
                segments: list[TrainingSegment] = []
                for message in messages:
                    if not isinstance(message, dict):
                        continue
                    role = message.get("role", "")
                    content = message.get("content", "")
                    segment = make_segment(role, content, cfg)
                    if segment is not None:
                        segments.append(segment)
                segments = compact_segments(segments, cfg)
                if segments:
                    yield segments, {
                        "dataset": "GEM/CrossWOZ",
                        "source_file": filename,
                        "dialog_id": str(dialog_id),
                        "type": dialog.get("type", ""),
                        "task_count": len(dialog.get("task description", [])) if isinstance(dialog.get("task description", []), list) else 0,
                    }


def iter_risawoz(cfg: LogicDialogConfig) -> Iterable[tuple[list[TrainingSegment], dict[str, Any]]]:
    local_dir = cfg.source_dir / "RiSAWOZ"
    split_files = {
        "train": local_dir / "train.json",
        "dev": local_dir / "dev.json",
        "test": local_dir / "test.json",
    }
    if all(path.exists() for path in split_files.values()):
        paths = split_files
    else:
        paths = {
            "train": hf_download("GEM/RiSAWOZ", "train.json"),
            "dev": hf_download("GEM/RiSAWOZ", "dev.json"),
            "test": hf_download("GEM/RiSAWOZ", "test.json"),
        }

    for split, filepath in paths.items():
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for dialog_index, dialog in enumerate(data):
            dialogue = dialog.get("dialogue", [])
            if not isinstance(dialogue, list) or not dialogue:
                continue
            segments: list[TrainingSegment] = []
            for turn in dialogue:
                if not isinstance(turn, dict):
                    continue
                user = make_segment("user", turn.get("user_utterance", ""), cfg)
                system = make_segment("assistant", turn.get("system_utterance", ""), cfg)
                if user is not None:
                    segments.append(user)
                if system is not None:
                    segments.append(system)
            if segments:
                yield segments, {
                    "dataset": "GEM/RiSAWOZ",
                    "source_file": f"{split}.json",
                    "dialogue_id": coerce_text(dialog.get("dialogue_id", "")),
                    "domains": dialog.get("domains", []),
                    "goal_len": len(coerce_text(dialog.get("goal", ""))),
                    "turn_count": len(dialogue),
                }


def write_source(
    f,
    *,
    source_name: str,
    target_records: int,
    examples: Iterable[tuple[list[TrainingSegment], dict[str, Any]]],
    tokenizer: HFWordPieceTokenizer,
    cfg: LogicDialogConfig,
    seen: set[str],
    seen_prompts: set[str],
    next_id: int,
    stats: dict[str, Any],
) -> int:
    if target_records <= 0:
        return next_id

    source_stats = stats["sources"].setdefault(
        source_name,
        {
            "target_records": target_records,
            "scanned_dialogs": 0,
            "candidate_records": 0,
            "written": 0,
            "skipped": 0,
            "duplicates": 0,
            "prompt_duplicates": 0,
            "skip_reasons": {},
        },
    )
    start = time.time()
    for raw_segments, meta in examples:
        if source_stats["written"] >= target_records:
            break
        source_stats["scanned_dialogs"] += 1
        stats["scanned_dialogs"] += 1

        records = list(records_from_dialog(raw_segments, cfg))
        source_stats["candidate_records"] += len(records)
        stats["candidate_records"] += len(records)
        if not records:
            source_stats["skipped"] += 1
            continue

        for segments in records:
            if source_stats["written"] >= target_records:
                break

            prompt_fingerprint = fingerprint_text(visible_prompt_key(segments))
            if cfg.dedupe_prompts and prompt_fingerprint in seen_prompts:
                source_stats["prompt_duplicates"] += 1
                stats["prompt_duplicates"] += 1
                continue

            fingerprint = fingerprint_text(record_key(segments))
            if fingerprint in seen:
                source_stats["duplicates"] += 1
                stats["duplicates"] += 1
                continue

            ok, reason, token_info = validate_record(segments, tokenizer, cfg)
            if not ok:
                source_stats["skip_reasons"][reason] = source_stats["skip_reasons"].get(reason, 0) + 1
                stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
                continue

            seen.add(fingerprint)
            seen_prompts.add(prompt_fingerprint)
            payload = make_record(
                f"{source_name}-{next_id:08d}",
                source_name,
                segments,
                {
                    **meta,
                    "mix_source": source_name,
                    "record_mode": cfg.record_mode,
                    "record_turns": len([segment for segment in segments if segment.role in {"user", "assistant"}]),
                },
                task="sft",
            )
            raw = json.dumps(payload, ensure_ascii=False)
            f.write(raw + "\n")
            next_id += 1
            source_stats["written"] += 1
            stats["written_records"] += 1
            stats["written_chars"] += len(raw)
            stats["token_sums"]["record"] += token_info["record_tokens"]
            stats["token_sums"]["prompt"] += token_info["prompt_tokens"]
            stats["token_sums"]["response"] += token_info["response_tokens"]
            if cfg.log_every > 0 and stats["written_records"] % cfg.log_every == 0:
                elapsed = time.time() - start
                print(
                    f"written={stats['written_records']} source={source_name} "
                    f"source_written={source_stats['written']} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    return next_id


def write_zh_logic_corpus(cfg: LogicDialogConfig = LOGIC_DIALOG_CFG) -> dict[str, Any]:
    if not cfg.tokenizer_json.exists():
        raise FileNotFoundError(f"tokenizer_json does not exist: {cfg.tokenizer_json}")
    tokenizer = HFWordPieceTokenizer.load(cfg.tokenizer_json)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cfg.out_path.with_name(cfg.out_path.name + ".tmp")
    meta_path = cfg.out_path.with_name(cfg.out_path.name + ".meta.json")

    stats: dict[str, Any] = {
        "schema": "tl-corpus-v1",
        "scanned_dialogs": 0,
        "candidate_records": 0,
        "written_records": 0,
        "written_chars": 0,
        "duplicates": 0,
        "prompt_duplicates": 0,
        "sources": {},
        "skip_reasons": {},
        "token_sums": {"record": 0, "prompt": 0, "response": 0},
    }
    seen: set[str] = set()
    seen_prompts: set[str] = set()
    next_id = 1
    start = time.time()

    with tmp_path.open("w", encoding="utf-8") as f:
        next_id = write_source(
            f,
            source_name="risawoz_zh_logic",
            target_records=cfg.risawoz_records,
            examples=iter_risawoz(cfg),
            tokenizer=tokenizer,
            cfg=cfg,
            seen=seen,
            seen_prompts=seen_prompts,
            next_id=next_id,
            stats=stats,
        )
        next_id = write_source(
            f,
            source_name="kdconv_zh_logic",
            target_records=cfg.kdconv_records,
            examples=iter_kdconv(cfg),
            tokenizer=tokenizer,
            cfg=cfg,
            seen=seen,
            seen_prompts=seen_prompts,
            next_id=next_id,
            stats=stats,
        )
        next_id = write_source(
            f,
            source_name="crosswoz_zh_logic",
            target_records=cfg.crosswoz_records,
            examples=iter_crosswoz(cfg),
            tokenizer=tokenizer,
            cfg=cfg,
            seen=seen,
            seen_prompts=seen_prompts,
            next_id=next_id,
            stats=stats,
        )
        next_id = write_source(
            f,
            source_name="duconv_zh_logic",
            target_records=cfg.duconv_records,
            examples=iter_duconv(cfg),
            tokenizer=tokenizer,
            cfg=cfg,
            seen=seen,
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
    parser = argparse.ArgumentParser(
        description="Prepare a Chinese logic-heavy dialogue corpus from grounded dialogue datasets."
    )
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--tokenizer-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--duconv-records", type=int, default=None)
    parser.add_argument("--crosswoz-records", type=int, default=None)
    parser.add_argument("--kdconv-records", type=int, default=None)
    parser.add_argument("--risawoz-records", type=int, default=None)
    parser.add_argument("--record-mode", default=None)
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
    parser.add_argument("--min-cjk-ratio", type=float, default=None)
    parser.add_argument("--max-ascii-letters", type=int, default=None)
    parser.add_argument("--allow-duplicate-prompts", action="store_true")
    return parser.parse_args()


def cfg_from_args(args: argparse.Namespace) -> LogicDialogConfig:
    updates: dict[str, Any] = {}
    for arg_name, field_name in (
        ("out_path", "out_path"),
        ("tokenizer_json", "tokenizer_json"),
        ("seed", "seed"),
        ("log_every", "log_every"),
        ("duconv_records", "duconv_records"),
        ("crosswoz_records", "crosswoz_records"),
        ("kdconv_records", "kdconv_records"),
        ("risawoz_records", "risawoz_records"),
        ("record_mode", "record_mode"),
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
        ("min_cjk_ratio", "min_cjk_ratio"),
        ("max_ascii_letters", "max_ascii_letters"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value
    if args.allow_duplicate_prompts:
        updates["dedupe_prompts"] = False
    return LogicDialogConfig(**{**asdict(LOGIC_DIALOG_CFG), **updates})


def main() -> None:
    write_zh_logic_corpus(cfg_from_args(parse_args()))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
