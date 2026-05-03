"""Merge the locally prepared dialogue and SFT corpora into one large mix.

This keeps the project on a single tl-corpus-v1 schema while letting training
use a larger, longer-context mix without touching the model code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

try:
    from .corpus import CORPUS_SCHEMA, record_from_payload, make_record
except ImportError:
    from corpus import CORPUS_SCHEMA, record_from_payload, make_record


@dataclass
class DialogueLogicMixConfig:
    out_path: Path = Path("data/corpus/dialogue_logic_mix_1024.tl.jsonl")
    log_every: int = 50000
    dedupe: bool = True
    source_paths: tuple[Path, ...] = (
        Path("data/corpus/zh_logic_dialog_mix.tl.jsonl"),
        Path("data/corpus/daily_dialog_expanded.tl.jsonl"),
        Path("data/corpus/short_dialog_mix.tl.jsonl"),
        Path("data/corpus/quality_sft_short.tl.jsonl"),
    )


DIALOGUE_LOGIC_MIX_CFG = DialogueLogicMixConfig()


def jsonable_config(cfg: DialogueLogicMixConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["source_paths"] = [str(path) for path in cfg.source_paths]
    payload["out_path"] = str(cfg.out_path)
    return payload


def iter_jsonl(path: Path) -> Iterable[tuple[int, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def canonical_record_payload(payload: Any, source: str, index: int) -> dict[str, Any]:
    record = record_from_payload(payload, source, index)
    return make_record(
        record.record_id,
        record.source,
        list(record.segments),
        record.meta,
        task=record.task,
        rejected_segments=list(record.rejected_segments),
    )


def fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main(cfg: DialogueLogicMixConfig = DIALOGUE_LOGIC_MIX_CFG) -> None:
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cfg.out_path.with_suffix(cfg.out_path.suffix + ".tmp")
    meta_path = cfg.out_path.with_name(cfg.out_path.name + ".meta.json")

    seen: set[str] = set()
    source_stats: dict[str, dict[str, int]] = {}
    written_records = 0
    written_chars = 0
    started = time.time()

    with tmp_path.open("w", encoding="utf-8") as out_f:
        for source_path in cfg.source_paths:
            if not source_path.exists():
                raise FileNotFoundError(f"source corpus does not exist: {source_path}")
            source_key = source_path.stem
            stats = source_stats.setdefault(source_key, {"input_records": 0, "written": 0, "duplicates": 0})
            for line_no, payload in iter_jsonl(source_path):
                stats["input_records"] += 1
                record_payload = canonical_record_payload(payload, str(source_path), line_no)
                fp = fingerprint(record_payload)
                if cfg.dedupe and fp in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(fp)
                out_f.write(json.dumps(record_payload, ensure_ascii=False) + "\n")
                stats["written"] += 1
                written_records += 1
                written_chars += len(json.dumps(record_payload, ensure_ascii=False))
                if written_records % cfg.log_every == 0:
                    elapsed = time.time() - started
                    print(
                        f"written_records={written_records} "
                        f"source={source_key} elapsed={elapsed:.1f}s"
                    )

    tmp_path.replace(cfg.out_path)
    meta = {
        "schema": CORPUS_SCHEMA,
        "config": jsonable_config(cfg),
        "stats": {
            "written_records": written_records,
            "written_chars": written_chars,
            "source_stats": source_stats,
            "elapsed_sec": time.time() - started,
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved merged corpus: {cfg.out_path} records={written_records}")


if __name__ == "__main__":
    main()
