"""Inspect the exact text windows used by text_lm.py training.

This script uses the same corpus reader, tokenizer cache, target mask, and
record-aware sampling rules as training. It is meant for checking whether the
model is actually seeing clean user/assistant text and which tokens contribute
to loss.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

try:
    from .config_loader import load_cfg, parse_bootstrap_args
    from .corpus import (
        IGNORE_INDEX,
        TrainingData,
        encode_or_load_training_data,
        load_or_build_tokenizer,
        read_corpus_records,
        render_record,
    )
    from .tokenizer import HFWordPieceTokenizer
except ImportError:
    from config_loader import load_cfg, parse_bootstrap_args
    from corpus import (
        IGNORE_INDEX,
        TrainingData,
        encode_or_load_training_data,
        load_or_build_tokenizer,
        read_corpus_records,
        render_record,
    )
    from tokenizer import HFWordPieceTokenizer


def parse_args() -> argparse.Namespace:
    bootstrap = parse_bootstrap_args(sys.argv[1:])
    parser = argparse.ArgumentParser(description="Print actual training text windows.")
    parser.add_argument("-c", "--config", default=bootstrap.config)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--record", type=int, default=None, help="0-based record index to inspect.")
    parser.add_argument("--window-start", type=int, default=None, help="Local token offset inside --record.")
    parser.add_argument("--max-input-chars", type=int, default=2200)
    parser.add_argument("--max-record-chars", type=int, default=2200)
    parser.add_argument("--token-table", type=int, default=80)
    parser.add_argument("--show-special", action="store_true")
    return parser.parse_args()


def shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... <truncated {len(text) - max_chars} chars>"


def decode_ids(tokenizer: HFWordPieceTokenizer, ids: np.ndarray | list[int], *, show_special: bool) -> str:
    return tokenizer.decode([int(v) for v in ids], skip_special=not show_special)


def supervised_chunks(tokenizer: HFWordPieceTokenizer, y: np.ndarray, *, show_special: bool) -> list[str]:
    chunks: list[str] = []
    current: list[int] = []
    for token_id in y:
        if int(token_id) == IGNORE_INDEX:
            if current:
                chunks.append(decode_ids(tokenizer, current, show_special=show_special))
                current = []
            continue
        current.append(int(token_id))
    if current:
        chunks.append(decode_ids(tokenizer, current, show_special=show_special))
    return chunks


def token_name(tokenizer: HFWordPieceTokenizer, token_id: int) -> str:
    if token_id == IGNORE_INDEX:
        return "<ignore>"
    token = tokenizer.id_to_token[int(token_id)]
    if token == "\n":
        return "\\n"
    return token.replace("\n", "\\n")


def print_token_table(tokenizer: HFWordPieceTokenizer, x: np.ndarray, y: np.ndarray, limit: int) -> None:
    if limit <= 0:
        return
    print("\n[token table]")
    print("idx | input_token -> target_token")
    print("----+----------------------------")
    for idx in range(min(limit, len(x))):
        target = token_name(tokenizer, int(y[idx]))
        flag = "train" if int(y[idx]) != IGNORE_INDEX else "ignore"
        print(f"{idx:>3} | {token_name(tokenizer, int(x[idx]))!r} -> {target!r} {flag}")
    if len(x) > limit:
        print(f"... <{len(x) - limit} more positions>")


def sample_stream_window(data: TrainingData, block_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(data.tokens) <= block_size + 1:
        raise ValueError("not enough tokens for the requested block size")
    start = int(rng.integers(0, len(data.tokens) - block_size))
    x = data.tokens[start : start + block_size]
    y = data.tokens[start + 1 : start + block_size + 1]
    if data.target_mask is not None:
        mask = data.target_mask[start + 1 : start + block_size + 1]
        y = np.where(mask, y, IGNORE_INDEX)
    return x.astype(np.int32), y.astype(np.int32), {"mode": "stream", "global_start": start}


def sample_record_window(
    data: TrainingData,
    block_size: int,
    rng: np.random.Generator,
    *,
    record_index: int | None = None,
    window_start: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if data.record_spans is None:
        return sample_stream_window(data, block_size, rng)

    for _ in range(1024):
        span_index = int(record_index) if record_index is not None else int(rng.integers(0, len(data.record_spans)))
        start, end = (int(v) for v in data.record_spans[span_index])
        length = end - start
        if length < 2:
            if record_index is not None:
                raise ValueError(f"record {record_index} is too short: {length} tokens")
            continue

        if length > block_size + 1:
            max_local_start = length - block_size - 1
            local_start = (
                min(max(int(window_start), 0), max_local_start)
                if window_start is not None
                else int(rng.integers(0, max_local_start + 1))
            )
            global_start = start + local_start
            x = data.tokens[global_start : global_start + block_size]
            y = data.tokens[global_start + 1 : global_start + block_size + 1]
            if data.target_mask is not None:
                mask = data.target_mask[global_start + 1 : global_start + block_size + 1]
                y = np.where(mask, y, IGNORE_INDEX)
            padded = 0
            valid = block_size
        else:
            local_start = 0
            valid = length - 1
            padded = block_size - valid
            x = np.full((block_size,), data.pad_id, dtype=np.int32)
            y = np.full((block_size,), IGNORE_INDEX, dtype=np.int32)
            x[:valid] = data.tokens[start : end - 1]
            targets = data.tokens[start + 1 : end]
            if data.target_mask is None:
                y[:valid] = targets
            else:
                mask = data.target_mask[start + 1 : end]
                y[:valid] = np.where(mask, targets, IGNORE_INDEX)
            global_start = start

        if np.any(y != IGNORE_INDEX):
            return (
                x.astype(np.int32, copy=False),
                y.astype(np.int32, copy=False),
                {
                    "mode": "record",
                    "record_index": span_index,
                    "record_start": start,
                    "record_end": end,
                    "record_tokens": length,
                    "local_start": local_start,
                    "global_start": global_start,
                    "valid_input_tokens": valid,
                    "padded_input_tokens": padded,
                },
            )
        if record_index is not None:
            raise ValueError(f"record {record_index} has no trainable targets in sampled window")

    raise ValueError("could not sample a window with at least one trainable target")


def print_window(
    index: int,
    x: np.ndarray,
    y: np.ndarray,
    info: dict[str, Any],
    records,
    tokenizer: HFWordPieceTokenizer,
    args: argparse.Namespace,
) -> None:
    train_targets = int(np.sum(y != IGNORE_INDEX))
    ignored_targets = int(np.sum(y == IGNORE_INDEX))
    pad_inputs = int(np.sum(x == tokenizer.pad_id))
    print("\n" + "=" * 88)
    print(f"sample {index}")
    print("info:", info)
    print(
        f"input_tokens={len(x)} train_targets={train_targets} "
        f"ignored_targets={ignored_targets} pad_inputs={pad_inputs}"
    )

    record_index = info.get("record_index")
    if record_index is not None and 0 <= int(record_index) < len(records):
        record = records[int(record_index)]
        print(f"\n[record {record_index}] id={record.record_id} source={record.source}")
        for seg_index, segment in enumerate(record.segments):
            print(
                f"  segment {seg_index}: role={segment.role} "
                f"train={segment.train} chars={len(segment.content)}"
            )
        print("\n[rendered record]")
        print(shorten(render_record(record), args.max_record_chars))

    print("\n[input context decoded]")
    print(shorten(decode_ids(tokenizer, x, show_special=args.show_special), args.max_input_chars))

    print("\n[supervised target chunks]")
    chunks = supervised_chunks(tokenizer, y, show_special=args.show_special)
    if not chunks:
        print("<none>")
    for chunk_index, chunk in enumerate(chunks):
        print(f"--- chunk {chunk_index} ---")
        print(shorten(chunk, args.max_input_chars))

    print_token_table(tokenizer, x, y, args.token_table)


def main() -> None:
    args = parse_args()
    cfg, _config_type, source = load_cfg(args.config, None, "debug_training_batch")
    if args.block_size is not None:
        cfg.block_size = args.block_size

    records, text, record_count, corpus_format = read_corpus_records(cfg)
    tokenizer = load_or_build_tokenizer(cfg, text)
    data = encode_or_load_training_data(cfg, tokenizer, text, records)
    seed = cfg.seed if args.seed is None else args.seed
    rng = np.random.default_rng(seed)

    print(f"config: {source}")
    print(f"corpus: {cfg.corpus} format={corpus_format} records={record_count}")
    print(f"loss_mode={cfg.loss_mode} block_size={cfg.block_size} record_aware={data.record_spans is not None}")
    print(f"tokens={len(data.tokens)} target_mask={data.target_mask is not None}")

    if args.record is not None:
        x, y, info = sample_record_window(
            data,
            cfg.block_size,
            rng,
            record_index=args.record,
            window_start=args.window_start,
        )
        print_window(0, x, y, info, records, tokenizer, args)
        return

    for sample_index in range(args.samples):
        x, y, info = sample_record_window(data, cfg.block_size, rng)
        print_window(sample_index, x, y, info, records, tokenizer, args)


if __name__ == "__main__":
    main()
