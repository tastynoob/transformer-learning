"""Corpus schema, rendering, tokenizer cache, and batch sampling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

import numpy as np

try:
    from .tokenizer import HFWordPieceTokenizer
except ImportError:
    from tokenizer import HFWordPieceTokenizer


CORPUS_SCHEMA = "tl-corpus-v1"
IGNORE_INDEX = -100
ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
}
ROLE_ALIASES = {
    "human": "user",
    "prompter": "user",
    "gpt": "assistant",
    "bot": "assistant",
    "text": "text",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
}
LEGACY_TURN_RE = re.compile(r"^(用户|User|Human|系统|System|助手|Assistant)\s*[：:]\s*(.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class TrainingSegment:
    role: str
    content: str
    train: bool


@dataclass(frozen=True)
class TrainingRecord:
    record_id: str
    source: str
    segments: tuple[TrainingSegment, ...]
    meta: dict[str, Any]


@dataclass(frozen=True)
class TrainingData:
    tokens: np.ndarray
    target_mask: np.ndarray | None = None
    record_spans: np.ndarray | None = None
    pad_id: int = 0


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


def normalize_role(role: Any) -> str:
    value = coerce_text(role).strip().lower()
    return ROLE_ALIASES.get(value, value)


def default_train_for_role(role: str) -> bool:
    return role in {"assistant", "text"}


def make_record(
    record_id: str,
    source: str,
    segments: list[dict[str, Any] | TrainingSegment],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload_segments: list[dict[str, Any]] = []
    for segment in segments:
        if isinstance(segment, TrainingSegment):
            role = segment.role
            content = segment.content
            train = segment.train
        else:
            role = normalize_role(segment.get("role", "text"))
            content = normalize_text(coerce_text(segment.get("content", "")))
            train = bool(segment.get("train", default_train_for_role(role)))
        if not content:
            continue
        payload_segments.append({"role": role, "content": content, "train": train})
    return {
        "schema": CORPUS_SCHEMA,
        "id": record_id,
        "source": source,
        "segments": payload_segments,
        "meta": meta or {},
    }


def record_from_payload(payload: Any, source: str, index: int, text_field: str = "text") -> TrainingRecord:
    if isinstance(payload, dict) and payload.get("schema") == CORPUS_SCHEMA:
        raw_segments = payload.get("segments", [])
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError(f"{source}:{index} has no non-empty segments")
        segments = tuple(_segment_from_payload(segment, f"{source}:{index}") for segment in raw_segments)
        segments = tuple(segment for segment in segments if segment.content)
        if not segments:
            raise ValueError(f"{source}:{index} has no non-empty segments")
        return TrainingRecord(
            record_id=coerce_text(payload.get("id", str(index))),
            source=coerce_text(payload.get("source", source)),
            segments=segments,
            meta=payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {},
        )

    if isinstance(payload, dict):
        if "input" in payload and "output" in payload:
            user = normalize_text(coerce_text(payload.get("input", "")))
            assistant = normalize_text(coerce_text(payload.get("output", "")))
            segments = [
                TrainingSegment("user", user, False),
                TrainingSegment("assistant", assistant, True),
            ]
            return TrainingRecord(str(index), source, tuple(segment for segment in segments if segment.content), {})
        if text_field not in payload:
            raise KeyError(f"{source}:{index} does not contain text_field={text_field!r}")
        text = strip_literal_special_tokens(normalize_text(coerce_text(payload[text_field])))
    else:
        text = strip_literal_special_tokens(normalize_text(coerce_text(payload)))

    if not text:
        raise ValueError(f"{source}:{index} is empty")
    return TrainingRecord(str(index), source, (TrainingSegment("text", text, True),), {"legacy_text": True})


def _segment_from_payload(segment: Any, source: str) -> TrainingSegment:
    if not isinstance(segment, dict):
        raise TypeError(f"{source} segment must be an object")
    role = normalize_role(segment.get("role", "text"))
    content = normalize_text(coerce_text(segment.get("content", "")))
    train = bool(segment.get("train", default_train_for_role(role)))
    if role not in {"system", "user", "assistant", "text"}:
        raise ValueError(f"{source} has unsupported role={role!r}")
    return TrainingSegment(role, content, train)


def strip_literal_special_tokens(text: str) -> str:
    return text.replace("<bos>", "").replace("<eos>", "").strip()


def parse_legacy_chat_text(text: str) -> list[TrainingSegment]:
    text = strip_literal_special_tokens(text)
    segments: list[TrainingSegment] = []
    current_role: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines, current_role
        if current_role is None:
            return
        content = normalize_text("\n".join(current_lines))
        if content:
            segments.append(TrainingSegment(current_role, content, default_train_for_role(current_role)))
        current_lines = []

    for line in text.splitlines():
        match = LEGACY_TURN_RE.match(line)
        if match:
            flush()
            label, content = match.groups()
            current_role = normalize_role(
                {
                    "用户": "user",
                    "human": "user",
                    "user": "user",
                    "系统": "system",
                    "system": "system",
                    "助手": "assistant",
                    "assistant": "assistant",
                }[label.lower()]
            )
            current_lines = [content]
        else:
            current_lines.append(line)
    flush()
    return segments


def render_segment(segment: TrainingSegment) -> str:
    if segment.role == "text":
        return segment.content
    label = ROLE_LABELS[segment.role]
    return f"{label}: {segment.content}\n"


def render_record(record: TrainingRecord) -> str:
    return "".join(render_segment(segment) for segment in record.segments).strip()


def segment_train_enabled(segment: TrainingSegment, loss_mode: str) -> bool:
    mode = loss_mode.lower()
    if mode == "all":
        return True
    if mode == "assistant":
        return segment.role == "assistant"
    if mode == "record":
        return segment.train
    raise ValueError("loss_mode must be one of: record, assistant, all")


def resolve_corpus_format(path: str | Path, corpus_format: str) -> str:
    fmt = corpus_format.strip().lower()
    if fmt == "auto":
        suffix = Path(path).suffix.lower()
        if suffix == ".jsonl":
            fmt = "jsonl"
        elif suffix == ".json":
            fmt = "json"
        else:
            fmt = "text"
    if fmt in {"txt", "plain"}:
        fmt = "text"
    if fmt == "ndjson":
        fmt = "jsonl"
    if fmt not in {"text", "jsonl", "json"}:
        raise ValueError(f"unsupported corpus_format={corpus_format!r}; expected auto, text, jsonl, or json")
    return fmt


def read_text(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def read_corpus_records(cfg) -> tuple[list[TrainingRecord], str, int, str]:
    corpus_format = resolve_corpus_format(cfg.corpus, cfg.corpus_format)
    text_field = getattr(cfg, "corpus_text_field", "text")
    if corpus_format == "text":
        text = normalize_text(read_text(cfg.corpus))
        records = [TrainingRecord("0", str(cfg.corpus), (TrainingSegment("text", text, True),), {})]
    elif corpus_format == "jsonl":
        records = []
        for line_no, line in enumerate(read_text(cfg.corpus).splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            records.append(record_from_payload(json.loads(line), str(cfg.corpus), line_no, text_field))
    else:
        payload = json.loads(read_text(cfg.corpus))
        if isinstance(payload, list):
            records = [record_from_payload(record, str(cfg.corpus), i, text_field) for i, record in enumerate(payload)]
        else:
            records = [record_from_payload(payload, str(cfg.corpus), 0, text_field)]

    if not records:
        raise ValueError(f"corpus is empty after reading {cfg.corpus}")
    if getattr(cfg, "max_chars", None) is not None:
        records = limit_records_by_chars(records, int(cfg.max_chars))
    text = getattr(cfg, "corpus_joiner", "\n").join(render_record(record) for record in records)
    return records, text, len(records), corpus_format


def limit_records_by_chars(records: list[TrainingRecord], max_chars: int) -> list[TrainingRecord]:
    if max_chars <= 0:
        return []
    out: list[TrainingRecord] = []
    used = 0
    for record in records:
        rendered = render_record(record)
        if used + len(rendered) <= max_chars:
            out.append(record)
            used += len(rendered)
            continue
        remaining = max_chars - used
        if remaining > 0:
            out.append(TrainingRecord(record.record_id, record.source, (TrainingSegment("text", rendered[:remaining], True),), record.meta))
        break
    return out


def save_tokenizer_copy(tokenizer: HFWordPieceTokenizer, path: Path, source: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(path)
    print(f"saved tokenizer copy: {path} source={source}")


def load_or_build_tokenizer(cfg, text: str) -> HFWordPieceTokenizer:
    if cfg.tokenizer_json.exists() and not cfg.rebuild_tokenizer:
        tokenizer = HFWordPieceTokenizer.load(cfg.tokenizer_json)
        if cfg.init_tokenizer_json is not None and cfg.init_tokenizer_json.exists():
            init_tokenizer = HFWordPieceTokenizer.load(cfg.init_tokenizer_json)
            if tokenizer_sha256(tokenizer) != tokenizer_sha256(init_tokenizer):
                raise ValueError(
                    f"tokenizer_json differs from init_tokenizer_json; refusing to mix token ids.\n"
                    f"tokenizer_json={cfg.tokenizer_json}\ninit_tokenizer_json={cfg.init_tokenizer_json}"
                )
        print(f"loaded tokenizer: {cfg.tokenizer_json} vocab_size={tokenizer.vocab_size}")
        return tokenizer

    if cfg.init_tokenizer_json is not None and not cfg.rebuild_tokenizer:
        if not cfg.init_tokenizer_json.exists():
            raise FileNotFoundError(f"init_tokenizer_json does not exist: {cfg.init_tokenizer_json}")
        tokenizer = HFWordPieceTokenizer.load(cfg.init_tokenizer_json)
        print(f"loaded init tokenizer: {cfg.init_tokenizer_json} vocab_size={tokenizer.vocab_size}")
        if cfg.tokenizer_json != cfg.init_tokenizer_json:
            save_tokenizer_copy(tokenizer, cfg.tokenizer_json, cfg.init_tokenizer_json)
        return tokenizer

    if not cfg.source_vocab.exists():
        raise FileNotFoundError(
            f"tokenizer JSON does not exist and source_vocab was not found: {cfg.source_vocab}. "
            "Download a HF WordPiece vocab.txt first, for example "
            "google-bert/bert-base-multilingual-cased/vocab.txt, or update source_vocab in init.py."
        )

    tokenizer = HFWordPieceTokenizer.from_vocab_file_and_corpus(
        cfg.source_vocab,
        [text],
        max_vocab_size=cfg.vocab_size,
        max_chinese_chars=cfg.max_chinese_chars,
        max_english_words=cfg.max_english_words,
        max_english_pieces=cfg.max_english_pieces,
        max_cjk_words=cfg.max_cjk_words,
        lowercase=cfg.lowercase,
    )
    cfg.tokenizer_json.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(cfg.tokenizer_json)
    print(f"saved tokenizer: {cfg.tokenizer_json} vocab_size={tokenizer.vocab_size}")
    return tokenizer


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenizer_sha256(tokenizer: HFWordPieceTokenizer) -> str:
    payload = {
        "lowercase": tokenizer.config.lowercase,
        "normalize": tokenizer.config.normalize,
        "max_input_chars_per_word": tokenizer.config.max_input_chars_per_word,
        "tokens": tokenizer.id_to_token,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(raw)


def token_cache_meta_path(token_cache: Path) -> Path:
    return token_cache.with_name(token_cache.name + ".meta.json")


def target_mask_cache_path(token_cache: Path) -> Path:
    return token_cache.with_name(token_cache.name + ".target_mask.npy")


def record_spans_cache_path(token_cache: Path) -> Path:
    return token_cache.with_name(token_cache.name + ".record_spans.npy")


def token_cache_metadata(cfg, tokenizer: HFWordPieceTokenizer, text: str, record_count: int) -> dict[str, Any]:
    return {
        "schema": CORPUS_SCHEMA,
        "corpus": str(cfg.corpus),
        "corpus_format": resolve_corpus_format(cfg.corpus, cfg.corpus_format),
        "corpus_text_field": getattr(cfg, "corpus_text_field", "text"),
        "corpus_joiner": getattr(cfg, "corpus_joiner", "\n"),
        "record_count": record_count,
        "loss_mode": getattr(cfg, "loss_mode", "record"),
        "record_aware_batches": bool(getattr(cfg, "record_aware_batches", True)),
        "text_chars": len(text),
        "text_sha256": sha256_text(text),
        "tokenizer_sha256": tokenizer_sha256(tokenizer),
        "tokenizer_vocab_size": tokenizer.vocab_size,
    }


def load_token_cache_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def encode_or_load_training_data(
    cfg,
    tokenizer: HFWordPieceTokenizer,
    text: str,
    records: list[TrainingRecord],
) -> TrainingData:
    metadata = token_cache_metadata(cfg, tokenizer, text, len(records))
    metadata_path = token_cache_meta_path(cfg.token_cache)
    mask_path = target_mask_cache_path(cfg.token_cache)
    spans_path = record_spans_cache_path(cfg.token_cache)
    record_aware_batches = bool(getattr(cfg, "record_aware_batches", True))
    if cfg.token_cache.exists():
        if cfg.retokenize:
            print(f"retokenize=True; rebuilding token cache: {cfg.token_cache}")
        else:
            cached_metadata = load_token_cache_metadata(metadata_path)
            if cached_metadata != metadata:
                print(f"token cache does not match current corpus/tokenizer; rebuilding: {cfg.token_cache}")
            else:
                tokens = np.load(cfg.token_cache).astype(np.int32, copy=False)
                target_mask = np.load(mask_path).astype(np.bool_, copy=False) if mask_path.exists() else None
                record_spans = None
                if record_aware_batches:
                    if not spans_path.exists():
                        print(f"record span cache missing; rebuilding token cache: {cfg.token_cache}")
                    else:
                        record_spans = np.load(spans_path).astype(np.int64, copy=False)
                        print(f"loaded token cache: {cfg.token_cache} tokens={len(tokens)}")
                        return TrainingData(tokens, target_mask, record_spans, tokenizer.pad_id)
                else:
                    print(f"loaded token cache: {cfg.token_cache} tokens={len(tokens)}")
                    return TrainingData(tokens, target_mask, None, tokenizer.pad_id)

    ids_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    offset = 0
    loss_mode = getattr(cfg, "loss_mode", "record")
    for record in records:
        ids, target_mask = encode_record(tokenizer, record, loss_mode)
        ids_parts.append(ids)
        mask_parts.append(target_mask)
        spans.append((offset, offset + len(ids)))
        offset += len(ids)
    ids = np.concatenate(ids_parts).astype(np.int32, copy=False)
    target_mask = np.concatenate(mask_parts).astype(np.bool_, copy=False)
    record_spans = np.asarray(spans, dtype=np.int64)
    compact_mask = None if bool(np.all(target_mask)) else target_mask

    cfg.token_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cfg.token_cache, ids)
    np.save(spans_path, record_spans)
    if compact_mask is None:
        if mask_path.exists():
            mask_path.unlink()
    else:
        np.save(mask_path, compact_mask)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"saved token cache: {cfg.token_cache} tokens={len(ids)} "
        f"records={len(record_spans)} masked_loss={compact_mask is not None}"
    )
    return TrainingData(ids, compact_mask, record_spans if record_aware_batches else None, tokenizer.pad_id)


def encode_record(tokenizer: HFWordPieceTokenizer, record: TrainingRecord, loss_mode: str) -> tuple[np.ndarray, np.ndarray]:
    ids: list[int] = [tokenizer.bos_id]
    target_mask: list[bool] = [False]
    last_train = False
    for segment in record.segments:
        rendered = render_segment(segment)
        segment_ids = tokenizer.encode(rendered, add_bos=False, add_eos=False, return_np=False)
        train = segment_train_enabled(segment, loss_mode)
        ids.extend(segment_ids)
        target_mask.extend([train] * len(segment_ids))
        last_train = train
    ids.append(tokenizer.eos_id)
    target_mask.append(last_train)
    return np.asarray(ids, dtype=np.int32), np.asarray(target_mask, dtype=np.bool_)


def random_batch(data: TrainingData, batch_size: int, block_size: int, rng: np.random.Generator):
    if data.record_spans is not None:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for _ in range(batch_size):
            x, y = _sample_record_window(data, block_size, rng)
            xs.append(x)
            ys.append(y)
        return np.stack(xs).astype(np.int32), np.stack(ys).astype(np.int32)

    tokens = data.tokens
    if len(tokens) <= block_size + 1:
        raise ValueError("not enough tokens for the requested block size")
    max_start = len(tokens) - block_size - 1
    starts = [_sample_start(max_start, block_size, rng, data.target_mask) for _ in range(batch_size)]
    x = np.stack([tokens[start : start + block_size] for start in starts])
    y = np.stack([tokens[start + 1 : start + block_size + 1] for start in starts])
    if data.target_mask is not None:
        masks = np.stack([data.target_mask[start + 1 : start + block_size + 1] for start in starts])
        y = np.where(masks, y, IGNORE_INDEX)
    return x.astype(np.int32), y.astype(np.int32)


def _sample_record_window(data: TrainingData, block_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    assert data.record_spans is not None
    for _ in range(1024):
        span_index = int(rng.integers(0, len(data.record_spans)))
        start, end = (int(v) for v in data.record_spans[span_index])
        length = end - start
        if length < 2:
            continue

        if length > block_size + 1:
            max_local_start = length - block_size - 1
            window_start = start + int(rng.integers(0, max_local_start + 1))
            x = data.tokens[window_start : window_start + block_size]
            y = data.tokens[window_start + 1 : window_start + block_size + 1]
            if data.target_mask is not None:
                mask = data.target_mask[window_start + 1 : window_start + block_size + 1]
                y = np.where(mask, y, IGNORE_INDEX)
        else:
            valid = length - 1
            x = np.full((block_size,), data.pad_id, dtype=np.int32)
            y = np.full((block_size,), IGNORE_INDEX, dtype=np.int32)
            x[:valid] = data.tokens[start : end - 1]
            targets = data.tokens[start + 1 : end]
            if data.target_mask is None:
                y[:valid] = targets
            else:
                mask = data.target_mask[start + 1 : end]
                y[:valid] = np.where(mask, targets, IGNORE_INDEX)

        if np.any(y != IGNORE_INDEX):
            return x.astype(np.int32, copy=False), y.astype(np.int32, copy=False)

    raise ValueError("could not sample a record window with at least one trainable target")


def _sample_start(
    max_start: int,
    block_size: int,
    rng: np.random.Generator,
    target_mask: np.ndarray | None,
) -> int:
    if target_mask is None:
        return int(rng.integers(0, max_start + 1))
    for _ in range(256):
        start = int(rng.integers(0, max_start + 1))
        if bool(np.any(target_mask[start + 1 : start + block_size + 1])):
            return start
    return int(rng.integers(0, max_start + 1))
