"""Convert local corpora into the project JSONL training schema.

Output schema, one JSON object per line:

{
  "schema": "tl-corpus-v1",
  "id": "example-000001",
  "source": "source name or path",
  "task": "lm | sft | preference",
  "segments": [
    {"role": "user", "content": "...", "train": false},
    {"role": "assistant", "content": "...", "train": true}
  ],
  "rejected_segments": [
    {"role": "user", "content": "...", "train": false},
    {"role": "assistant", "content": "bad answer", "train": true}
  ],
  "meta": {}
}

Roles are system, user, assistant, or text. During training, loss_mode=record
uses each segment's train flag; text/pretrain segments train all tokens, while
chat/SFT records normally train assistant tokens only. rejected_segments is
optional and is ignored by plain LM/SFT training; preference trainers can use it
without requiring a second corpus schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Iterable

try:
    from .corpus import (
        CORPUS_SCHEMA,
        TrainingSegment,
        coerce_text,
        make_record,
        normalize_role,
        normalize_text,
        parse_legacy_chat_text,
        record_from_payload,
        strip_literal_special_tokens,
    )
except ImportError:
    from corpus import (
        CORPUS_SCHEMA,
        TrainingSegment,
        coerce_text,
        make_record,
        normalize_role,
        normalize_text,
        parse_legacy_chat_text,
        record_from_payload,
        strip_literal_special_tokens,
    )


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def iter_jsonl(path: Path) -> Iterable[tuple[int, Any]]:
    for line_no, line in enumerate(read_text(path).splitlines(), start=1):
        line = line.strip()
        if line:
            yield line_no, json.loads(line)


def message_content(message: dict[str, Any]) -> str:
    return coerce_text(message.get("content", message.get("value", "")))


def message_role(message: dict[str, Any]) -> str:
    return normalize_role(message.get("role", message.get("from", "")))


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
        content = normalize_text(message_content(message))
        if content:
            segments.append(TrainingSegment(role, content, role == "assistant"))
    return segments


def segments_from_prompt(value: Any, args: argparse.Namespace) -> list[TrainingSegment]:
    if isinstance(value, list):
        return segments_from_messages(value)
    if isinstance(value, dict):
        if args.messages_field in value:
            return segments_from_messages(value.get(args.messages_field))
        if "messages" in value:
            return segments_from_messages(value.get("messages"))
        if "conversations" in value:
            return segments_from_messages(value.get("conversations"))
        if "role" in value or "from" in value:
            return segments_from_messages([value])
        if args.text_field in value:
            value = value.get(args.text_field)
    content = normalize_text(coerce_text(value))
    return [TrainingSegment("user", content, False)] if content else []


def segments_from_response(value: Any, args: argparse.Namespace) -> list[TrainingSegment]:
    if isinstance(value, list):
        return segments_from_messages(value)
    if isinstance(value, dict):
        if args.messages_field in value:
            return segments_from_messages(value.get(args.messages_field))
        if "messages" in value:
            return segments_from_messages(value.get("messages"))
        if "conversations" in value:
            return segments_from_messages(value.get("conversations"))
        if "role" in value or "from" in value:
            return segments_from_messages([value])
        if args.output_field in value:
            value = value.get(args.output_field)
        elif args.text_field in value:
            value = value.get(args.text_field)
    content = normalize_text(coerce_text(value))
    return [TrainingSegment("assistant", content, True)] if content else []


def compose_preference_branch(prompt_segments: list[TrainingSegment], response_segments: list[TrainingSegment]) -> list[TrainingSegment]:
    if not prompt_segments:
        return response_segments
    if response_segments and response_segments[0].role in {"system", "user"}:
        return response_segments
    return prompt_segments + response_segments


def segments_from_instruction(record: dict[str, Any], args: argparse.Namespace) -> list[TrainingSegment]:
    instruction = normalize_text(coerce_text(record.get(args.instruction_field, "")))
    input_text = normalize_text(coerce_text(record.get(args.input_field, "")))
    output = normalize_text(coerce_text(record.get(args.output_field, "")))
    if not instruction or not output:
        return []
    user = instruction if not input_text else f"{instruction}\n{input_text}"
    return [TrainingSegment("user", user, False), TrainingSegment("assistant", output, True)]


def segments_from_text(text: str, parse_chat_labels: bool) -> list[TrainingSegment]:
    text = strip_literal_special_tokens(normalize_text(text))
    if not text:
        return []
    if parse_chat_labels:
        parsed = parse_legacy_chat_text(text)
        roles = {segment.role for segment in parsed}
        if parsed and ("user" in roles or "assistant" in roles):
            return parsed
    return [TrainingSegment("text", text, True)]


def truncate_segments(segments: list[TrainingSegment], max_chars: int | None) -> list[TrainingSegment]:
    if max_chars is None:
        return segments
    used = 0
    out: list[TrainingSegment] = []
    for segment in segments:
        if used >= max_chars:
            break
        remaining = max_chars - used
        content = segment.content[:remaining].rstrip()
        if content:
            out.append(TrainingSegment(segment.role, content, segment.train))
            used += len(content)
    return out


def chunk_text(text: str, chunk_chars: int) -> Iterable[str]:
    text = strip_literal_special_tokens(normalize_text(text))
    if not text:
        return
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        extra = len(paragraph) + (2 if current else 0)
        if current and current_len + extra > chunk_chars:
            yield "\n\n".join(current)
            current = []
            current_len = 0
        if len(paragraph) > chunk_chars:
            for start in range(0, len(paragraph), chunk_chars):
                piece = paragraph[start : start + chunk_chars].strip()
                if piece:
                    yield piece
            continue
        current.append(paragraph)
        current_len += extra
    if current:
        yield "\n\n".join(current)


def convert_payload(payload: Any, line_no: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    source = args.source or str(args.input)
    record_id = f"{args.id_prefix}-{line_no:06d}"
    if isinstance(payload, dict) and payload.get("schema") == CORPUS_SCHEMA:
        record = record_from_payload(payload, source, line_no, args.text_field)
        return [
            make_record(
                record.record_id,
                record.source,
                list(record.segments),
                record.meta,
                task=record.task,
                rejected_segments=list(record.rejected_segments),
            )
        ]

    if isinstance(payload, dict) and "chosen" in payload and "rejected" in payload:
        prompt = payload.get("prompt", payload.get(args.input_field, payload.get(args.messages_field)))
        prompt_segments = segments_from_prompt(prompt, args)
        chosen_segments = segments_from_response(payload.get("chosen"), args)
        rejected_segments = segments_from_response(payload.get("rejected"), args)
        positive = compose_preference_branch(prompt_segments, chosen_segments)
        negative = compose_preference_branch(prompt_segments, rejected_segments)
        positive = truncate_segments(positive, args.max_chars_per_record)
        negative = truncate_segments(negative, args.max_chars_per_record)
        if positive and negative:
            return [
                make_record(
                    record_id,
                    source,
                    positive,
                    {"source_line": line_no},
                    task="preference",
                    rejected_segments=negative,
                )
            ]
        return []

    segments: list[TrainingSegment] = []
    if isinstance(payload, dict):
        if args.messages_field in payload:
            segments = segments_from_messages(payload.get(args.messages_field))
        elif "conversations" in payload:
            segments = segments_from_messages(payload.get("conversations"))
        elif args.input_field in payload and args.output_field in payload:
            segments = segments_from_instruction(payload, args)
        elif args.text_field in payload:
            segments = segments_from_text(coerce_text(payload.get(args.text_field)), args.parse_chat_labels)
    else:
        segments = segments_from_text(coerce_text(payload), args.parse_chat_labels)

    segments = truncate_segments(segments, args.max_chars_per_record)
    if not segments:
        return []
    return [
        make_record(
            record_id,
            source,
            list(segments),
            {"source_line": line_no},
        )
    ]


def convert_text_file(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []
    for i, chunk in enumerate(chunk_text(read_text(args.input), args.chunk_chars), start=1):
        records.append(
            make_record(
                f"{args.id_prefix}-{i:06d}",
                args.source or str(args.input),
                [TrainingSegment("text", chunk, True)],
                {"source_chunk": i},
            )
        )
        if args.max_records is not None and len(records) >= args.max_records:
            break
    return records


def iter_converted_records(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    kind = args.kind
    if kind == "auto":
        kind = "text" if args.input.suffix.lower() in {".txt", ".md"} else "jsonl"
    if kind == "text":
        yield from convert_text_file(args)
        return
    if kind != "jsonl":
        raise ValueError("kind must be auto, text, or jsonl")

    written = 0
    for line_no, payload in iter_jsonl(args.input):
        for record in convert_payload(payload, line_no, args):
            yield record
            written += 1
            if args.max_records is not None and written >= args.max_records:
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local corpora to tl-corpus-v1 JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("auto", "text", "jsonl"), default="auto")
    parser.add_argument("--source", default=None)
    parser.add_argument("--id-prefix", default="example")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument("--instruction-field", default="instruction")
    parser.add_argument("--input-field", default="input")
    parser.add_argument("--output-field", default="output")
    parser.add_argument("--chunk-chars", type=int, default=4000)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-chars-per-record", type=int, default=None)
    parser.add_argument("--parse-chat-labels", dest="parse_chat_labels", action="store_true", default=True)
    parser.add_argument("--no-parse-chat-labels", dest="parse_chat_labels", action="store_false")
    parser.add_argument("--log-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_name(args.output.name + ".tmp")
    meta_path = args.output.with_name(args.output.name + ".meta.json")
    start = time.time()
    stats = {"schema": CORPUS_SCHEMA, "written_records": 0, "written_chars": 0}

    with tmp_path.open("w", encoding="utf-8") as f:
        for record in iter_converted_records(args):
            raw = json.dumps(record, ensure_ascii=False)
            f.write(raw + "\n")
            stats["written_records"] += 1
            stats["written_chars"] += len(raw)
            if args.log_every > 0 and stats["written_records"] % args.log_every == 0:
                elapsed = time.time() - start
                print(f"written={stats['written_records']} chars={stats['written_chars']} elapsed={elapsed:.1f}s")

    tmp_path.replace(args.output)
    stats["elapsed_sec"] = round(time.time() - start, 3)
    meta = {
        "input": str(args.input),
        "output": str(args.output),
        "kind": args.kind,
        "stats": stats,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved corpus: {args.output}")
    print(f"saved metadata: {meta_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
