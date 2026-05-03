"""Build a short-dialogue SFT corpus in tl-corpus-v1 format.

This script focuses on short, local-context conversations. Long source
dialogues are converted into sliding windows ending at an assistant turn, so
the target answer is not silently truncated away by the trainer.

Default sources:
- silver/lccc
- facebook/empathetic_dialogues raw archive
- awsaf49/persona-chat
- google/Synthetic-Persona-Chat
- anezatra/blended-skill-talk
- Cornell Movie Dialogs raw archive
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tarfile
import time
from typing import Any, Iterable
import unicodedata
from urllib.request import urlretrieve
import zipfile

try:
    from .corpus import TrainingSegment, coerce_text, make_record, normalize_role
except ImportError:
    from corpus import TrainingSegment, coerce_text, make_record, normalize_role


EMPATHETIC_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
CORNELL_URL = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"


@dataclass
class ShortDialogConfig:
    out_path: Path = Path("data/corpus/short_dialog_en_mix.tl.jsonl")
    raw_dir: Path = Path("data/raw/short_dialog")
    seed: int = 0
    log_every: int = 5000

    lccc_repo: str = "silver/lccc"
    lccc_files: str = "lccc_base_train.jsonl.gz"
    lccc_records: int = 0

    empathetic_records: int = 30000
    empathetic_splits: str = "train,valid,test"

    persona_repo: str = "awsaf49/persona-chat"
    persona_files: str = "train_both_original.txt,train_both_revised.txt"
    persona_records: int = 30000

    synthetic_persona_repo: str = "google/Synthetic-Persona-Chat"
    synthetic_persona_files: str = "data/Synthetic-Persona-Chat_train.csv"
    synthetic_persona_records: int = 30000

    blended_skill_repo: str = "anezatra/blended-skill-talk"
    blended_skill_files: str = "data/train-00000-of-00001.parquet,data/validation-00000-of-00001.parquet"
    blended_skill_records: int = 20000

    cornell_records: int = 30000

    min_turns: int = 2
    max_turns: int = 4
    min_record_chars: int = 8
    max_record_chars: int = 1600
    min_assistant_chars: int = 2
    max_assistant_chars: int = 600
    target_last_assistant: bool = True
    dedupe_prompts: bool = True
    shuffle_source_order: bool = True


SHORT_DIALOG_CFG = ShortDialogConfig()


def require_hf_download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Missing dependency: huggingface_hub. Install it before preparing HF corpora.") from exc
    return hf_hub_download


def require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Missing dependency: pandas/pyarrow is required for BlendedSkillTalk parquet files.") from exc
    return pd


def csv_parts(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def clean_content(value: Any) -> str:
    text = unicodedata.normalize("NFC", coerce_text(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("_comma_", ",")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def make_segment(role: str, content: Any) -> TrainingSegment | None:
    role = normalize_role(role)
    if role not in {"system", "user", "assistant"}:
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
            merged = out[-1].content + "\n" + normalized.content
            out[-1] = TrainingSegment(normalized.role, merged, normalized.train)
        else:
            out.append(normalized)
    return out


def alternating_segments(turns: Iterable[Any], first_role: str = "user") -> list[TrainingSegment]:
    roles = ("user", "assistant") if first_role == "user" else ("assistant", "user")
    segments: list[TrainingSegment] = []
    for index, turn in enumerate(turns):
        segment = make_segment(roles[index % 2], turn)
        if segment is not None:
            segments.append(segment)
    return compact_segments(segments)


def retarget_last_assistant(segments: list[TrainingSegment]) -> list[TrainingSegment]:
    last_assistant = max((i for i, segment in enumerate(segments) if segment.role == "assistant"), default=-1)
    return [
        TrainingSegment(segment.role, segment.content, segment.role == "assistant" and i == last_assistant)
        for i, segment in enumerate(segments)
    ]


def valid_window(segments: list[TrainingSegment], cfg: ShortDialogConfig) -> bool:
    if not segments or segments[-1].role != "assistant":
        return False
    turns = [segment for segment in segments if segment.role in {"user", "assistant"}]
    if len(turns) < cfg.min_turns:
        return False
    if not any(segment.role == "user" for segment in turns):
        return False
    total_chars = sum(len(segment.content) for segment in segments)
    if total_chars < cfg.min_record_chars or total_chars > cfg.max_record_chars:
        return False
    target_chars = sum(len(segment.content) for segment in segments if segment.train)
    if target_chars < cfg.min_assistant_chars or target_chars > cfg.max_assistant_chars:
        return False
    if any(len(segment.content) > cfg.max_assistant_chars for segment in segments if segment.role == "assistant"):
        return False
    return True


def windows_from_dialog(segments: list[TrainingSegment], cfg: ShortDialogConfig) -> list[list[TrainingSegment]]:
    compacted = [segment for segment in compact_segments(segments) if segment.role in {"user", "assistant"}]
    windows: list[list[TrainingSegment]] = []
    for index, segment in enumerate(compacted):
        if segment.role != "assistant":
            continue
        start = max(0, index - cfg.max_turns + 1)
        while start <= index and compacted[start].role == "assistant":
            start += 1
        window = compacted[start : index + 1]
        if cfg.target_last_assistant:
            window = retarget_last_assistant(window)
        if valid_window(window, cfg):
            windows.append(window)
    return windows


def fingerprint_segments(segments: list[TrainingSegment]) -> str:
    normalized = "\n".join(
        f"{segment.role}:{' '.join(segment.content.lower().split())}:{int(segment.train)}"
        for segment in segments
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def fingerprint_prompt(segments: list[TrainingSegment]) -> str:
    normalized = "\n".join(
        f"{segment.role}:{' '.join(segment.content.lower().split())}"
        for segment in segments
        if not segment.train
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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


def iter_lccc(cfg: ShortDialogConfig) -> Iterable[tuple[list[TrainingSegment], dict[str, Any]]]:
    for filename in csv_parts(cfg.lccc_files):
        path = hf_download(cfg.lccc_repo, filename)
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
                    yield alternating_segments(dialog), {
                        "dataset": cfg.lccc_repo,
                        "file": filename,
                        "source_index": line_no,
                    }


def iter_empathetic(cfg: ShortDialogConfig) -> Iterable[tuple[list[TrainingSegment], dict[str, Any]]]:
    archive_path = download_external(EMPATHETIC_URL, cfg.raw_dir, "empatheticdialogues.tar.gz")
    wanted = {f"empatheticdialogues/{split}.csv" for split in csv_parts(cfg.empathetic_splits)}
    grouped: dict[str, list[dict[str, str]]] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.name not in wanted:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            reader = csv.DictReader((line.decode("utf-8") for line in extracted))
            for row in reader:
                conv_id = row.get("conv_id", "")
                grouped.setdefault(f"{member.name}:{conv_id}", []).append(row)

    for index, (conv_key, rows) in enumerate(grouped.items()):
        rows.sort(key=lambda row: int(row.get("utterance_idx", 0)))
        speaker_roles: dict[str, str] = {}
        segments: list[TrainingSegment] = []
        for row in rows:
            speaker = row.get("speaker_idx", "")
            if speaker not in speaker_roles:
                speaker_roles[speaker] = "user" if len(speaker_roles) == 0 else "assistant"
            segment = make_segment(speaker_roles[speaker], row.get("utterance", ""))
            if segment is not None:
                segments.append(segment)
        yield compact_segments(segments), {
            "dataset": "facebook/empathetic_dialogues",
            "conversation_key": conv_key,
            "source_index": index,
            "context": rows[0].get("context") if rows else None,
            "prompt": rows[0].get("prompt") if rows else None,
        }


PERSONA_PREFIXES = ("your persona:", "partner's persona:")


def iter_persona_chat(cfg: ShortDialogConfig) -> Iterable[tuple[list[TrainingSegment], dict[str, Any]]]:
    for filename in csv_parts(cfg.persona_files):
        path = hf_download(cfg.persona_repo, filename)
        personas: list[str] = []
        conversation_index = 0
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                if " " not in line:
                    continue
                turn_no, payload = line.split(" ", 1)
                if turn_no == "1":
                    conversation_index += 1
                    personas = []
                lowered = payload.lower()
                if lowered.startswith(PERSONA_PREFIXES):
                    personas.append(clean_content(payload.split(":", 1)[1]))
                    continue
                fields = payload.split("\t")
                if len(fields) < 2:
                    continue
                user = clean_content(fields[0])
                assistant = clean_content(fields[1])
                if user and assistant:
                    yield alternating_segments([user, assistant]), {
                        "dataset": cfg.persona_repo,
                        "file": filename,
                        "source_line": line_no,
                        "conversation_index": conversation_index,
                        "personas": personas[:],
                    }


USER_LINE_RE = re.compile(r"^User\s*([12])\s*:\s*(.*)$", re.IGNORECASE)


def parse_user_labeled_conversation(text: str) -> list[TrainingSegment]:
    segments: list[TrainingSegment] = []
    for raw_line in coerce_text(text).splitlines():
        match = USER_LINE_RE.match(raw_line.strip())
        if not match:
            continue
        role = "user" if match.group(1) == "1" else "assistant"
        segment = make_segment(role, match.group(2))
        if segment is not None:
            segments.append(segment)
    return compact_segments(segments)


def iter_synthetic_persona(cfg: ShortDialogConfig) -> Iterable[tuple[list[TrainingSegment], dict[str, Any]]]:
    for filename in csv_parts(cfg.synthetic_persona_files):
        path = hf_download(cfg.synthetic_persona_repo, filename)
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for index, row in enumerate(reader):
                yield parse_user_labeled_conversation(row.get("Best Generated Conversation", "")), {
                    "dataset": cfg.synthetic_persona_repo,
                    "file": filename,
                    "source_index": index,
                    "user_1_personas": row.get("user 1 personas", ""),
                    "user_2_personas": row.get("user 2 personas", ""),
                }


def pylist(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return [value]


def iter_blended_skill(cfg: ShortDialogConfig) -> Iterable[tuple[list[TrainingSegment], dict[str, Any]]]:
    pd = require_pandas()
    for filename in csv_parts(cfg.blended_skill_files):
        path = hf_download(cfg.blended_skill_repo, filename)
        df = pd.read_parquet(path)
        for index, row in df.iterrows():
            segments: list[TrainingSegment] = []
            previous = pylist(row.get("previous_utterance"))
            for i, utterance in enumerate(previous):
                segment = make_segment("user" if i % 2 == 0 else "assistant", utterance)
                if segment is not None:
                    segments.append(segment)
            free_messages = pylist(row.get("free_messages"))
            guided_messages = pylist(row.get("guided_messages"))
            for user_text, assistant_text in zip(free_messages, guided_messages):
                user_segment = make_segment("user", user_text)
                assistant_segment = make_segment("assistant", assistant_text)
                if user_segment is not None:
                    segments.append(user_segment)
                if assistant_segment is not None:
                    segments.append(assistant_segment)
            yield compact_segments(segments), {
                "dataset": cfg.blended_skill_repo,
                "file": filename,
                "source_index": int(index),
                "context": clean_content(row.get("context", "")),
                "additional_context": clean_content(row.get("additional_context", "")),
            }


def read_zip_text(archive: zipfile.ZipFile, name: str, encoding: str = "latin-1") -> str:
    with archive.open(name) as f:
        return f.read().decode(encoding)


def iter_cornell(cfg: ShortDialogConfig) -> Iterable[tuple[list[TrainingSegment], dict[str, Any]]]:
    zip_path = download_external(CORNELL_URL, cfg.raw_dir, "cornell_movie_dialogs_corpus.zip")
    base = "cornell movie-dialogs corpus"
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        prefix = base if f"{base}/movie_lines.txt" in names else f"cornell movie-dialogs corpus/{base}"
        lines_text = read_zip_text(archive, f"{prefix}/movie_lines.txt")
        conversations_text = read_zip_text(archive, f"{prefix}/movie_conversations.txt")

    line_by_id: dict[str, str] = {}
    for raw_line in lines_text.splitlines():
        parts = raw_line.split(" +++$+++ ", 4)
        if len(parts) >= 5:
            line_by_id[parts[0].strip()] = clean_content(parts[4])

    for index, raw_line in enumerate(conversations_text.splitlines()):
        parts = raw_line.split(" +++$+++ ", 3)
        if len(parts) < 4:
            continue
        try:
            line_ids = ast.literal_eval(parts[3].strip())
        except (SyntaxError, ValueError):
            continue
        turns = [line_by_id.get(line_id, "") for line_id in line_ids]
        yield alternating_segments(turns), {
            "dataset": "cornell-movie-dialog/cornell_movie_dialog",
            "source_index": index,
            "movie_id": parts[2].strip(),
            "line_ids": line_ids,
        }


def write_source(
    f,
    *,
    source_name: str,
    target_records: int,
    examples: Iterable[tuple[list[TrainingSegment], dict[str, Any]]],
    cfg: ShortDialogConfig,
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
            "candidate_windows": 0,
            "written": 0,
            "skipped": 0,
            "duplicates": 0,
            "prompt_duplicates": 0,
        },
    )
    start = time.time()
    for raw_segments, meta in examples:
        if source_stats["written"] >= target_records:
            break
        source_stats["scanned_dialogs"] += 1
        windows = windows_from_dialog(raw_segments, cfg)
        source_stats["candidate_windows"] += len(windows)
        if not windows:
            source_stats["skipped"] += 1
            continue
        for segments in windows:
            if source_stats["written"] >= target_records:
                break
            prompt_fingerprint = fingerprint_prompt(segments)
            if cfg.dedupe_prompts and prompt_fingerprint in seen_prompts:
                source_stats["prompt_duplicates"] += 1
                continue
            fingerprint = fingerprint_segments(segments)
            if fingerprint in seen:
                source_stats["duplicates"] += 1
                continue
            seen.add(fingerprint)
            seen_prompts.add(prompt_fingerprint)
            payload = make_record(
                f"shortdialog-{next_id:07d}",
                source_name,
                segments,
                {
                    **meta,
                    "mix_source": source_name,
                    "window_turns": len([segment for segment in segments if segment.role in {"user", "assistant"}]),
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


def jsonable_config(cfg: ShortDialogConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def source_specs(cfg: ShortDialogConfig):
    specs = [
        ("lccc", cfg.lccc_records, lambda: iter_lccc(cfg)),
        ("empathetic_dialogues", cfg.empathetic_records, lambda: iter_empathetic(cfg)),
        ("persona_chat", cfg.persona_records, lambda: iter_persona_chat(cfg)),
        ("synthetic_persona_chat", cfg.synthetic_persona_records, lambda: iter_synthetic_persona(cfg)),
        ("blended_skill_talk", cfg.blended_skill_records, lambda: iter_blended_skill(cfg)),
        ("cornell_movie_dialog", cfg.cornell_records, lambda: iter_cornell(cfg)),
    ]
    if cfg.shuffle_source_order:
        rng = random.Random(cfg.seed)
        rng.shuffle(specs)
    return specs


def write_short_dialog_mix(cfg: ShortDialogConfig = SHORT_DIALOG_CFG) -> dict[str, Any]:
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
    seen_prompts: set[str] = set()
    next_id = 1
    start = time.time()

    with tmp_path.open("w", encoding="utf-8") as f:
        for source_name, target_records, examples_factory in source_specs(cfg):
            print(f"source={source_name} target_records={target_records}")
            next_id = write_source(
                f,
                source_name=source_name,
                target_records=target_records,
                examples=examples_factory(),
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
    parser = argparse.ArgumentParser(description="Prepare a mixed short-dialogue tl-corpus-v1 JSONL corpus.")
    parser.add_argument("--out-path", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--lccc-records", type=int, default=None)
    parser.add_argument("--empathetic-records", type=int, default=None)
    parser.add_argument("--persona-records", type=int, default=None)
    parser.add_argument("--synthetic-persona-records", type=int, default=None)
    parser.add_argument("--blended-skill-records", type=int, default=None)
    parser.add_argument("--cornell-records", type=int, default=None)
    parser.add_argument("--min-turns", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--min-record-chars", type=int, default=None)
    parser.add_argument("--max-record-chars", type=int, default=None)
    parser.add_argument("--min-assistant-chars", type=int, default=None)
    parser.add_argument("--max-assistant-chars", type=int, default=None)
    parser.add_argument("--ordered-sources", action="store_true")
    parser.add_argument("--train-all-assistant-turns", action="store_true")
    parser.add_argument("--allow-duplicate-prompts", action="store_true")
    return parser.parse_args()


def cfg_from_args(args: argparse.Namespace) -> ShortDialogConfig:
    updates: dict[str, Any] = {}
    for arg_name, field_name in (
        ("out_path", "out_path"),
        ("raw_dir", "raw_dir"),
        ("seed", "seed"),
        ("log_every", "log_every"),
        ("lccc_records", "lccc_records"),
        ("empathetic_records", "empathetic_records"),
        ("persona_records", "persona_records"),
        ("synthetic_persona_records", "synthetic_persona_records"),
        ("blended_skill_records", "blended_skill_records"),
        ("cornell_records", "cornell_records"),
        ("min_turns", "min_turns"),
        ("max_turns", "max_turns"),
        ("min_record_chars", "min_record_chars"),
        ("max_record_chars", "max_record_chars"),
        ("min_assistant_chars", "min_assistant_chars"),
        ("max_assistant_chars", "max_assistant_chars"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value
    if args.ordered_sources:
        updates["shuffle_source_order"] = False
    if args.train_all_assistant_turns:
        updates["target_last_assistant"] = False
    if args.allow_duplicate_prompts:
        updates["dedupe_prompts"] = False
    return ShortDialogConfig(**{**asdict(SHORT_DIALOG_CFG), **updates})


def main() -> None:
    write_short_dialog_mix(cfg_from_args(parse_args()))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
