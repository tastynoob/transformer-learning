"""Small tokenizer built from a pruned Hugging Face WordPiece vocabulary.

Use a HF vocab as the source of truth, then keep only a small subset for a toy
transformer. A good source is a BERT-style `vocab.txt`, for example
`bert-base-multilingual-cased` or `bert-base-chinese`.

This module intentionally does not depend on `transformers` at runtime. It
loads `vocab.txt`/WordPiece `tokenizer.json`, builds a compact vocab, and uses
standard greedy WordPiece matching.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import re
import unicodedata
from typing import Iterable, Sequence

import numpy as np


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPACE_TOKEN = "<space>"
NEWLINE_TOKEN = "<newline>"
TAB_TOKEN = "<tab>"

SPECIAL_TOKENS = (
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    SPACE_TOKEN,
    NEWLINE_TOKEN,
    TAB_TOKEN,
)
CONTROL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)

HF_SPECIAL_RE = re.compile(r"^\[(?:PAD|UNK|CLS|SEP|MASK|unused\d+)\]$")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

ASCII_FALLBACK = tuple("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
COMMON_PUNCTUATION = tuple(
    ".,!?;:'\"()[]{}<>+-=*/\\|_@#$%^&`~，。！？；：、（）《》【】“”‘’"
)


@dataclass(frozen=True)
class TokenizerConfig:
    lowercase: bool = True
    normalize: str = "NFC"
    max_input_chars_per_word: int = 100


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _is_cjk_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _contains_cjk(token: str) -> bool:
    return any(_is_cjk_char(ch) for ch in token)


def _is_ascii_wordpiece(token: str) -> bool:
    piece = token[2:] if token.startswith("##") else token
    return bool(piece) and all(ch.isascii() and (ch.isalnum() or ch == "'") for ch in piece)


def _is_whole_ascii_word(token: str) -> bool:
    return not token.startswith("##") and _is_ascii_wordpiece(token) and any(ch.isalpha() for ch in token)


def _is_ascii_continuation(token: str) -> bool:
    return token.startswith("##") and _is_ascii_wordpiece(token)


def _is_single_cjk(token: str) -> bool:
    return len(token) == 1 and _is_cjk_char(token)


def _is_supported_source_token(token: str) -> bool:
    if not token or token.isspace() or HF_SPECIAL_RE.match(token):
        return False
    if token.startswith("[") and token.endswith("]"):
        return False
    return True


def _is_punctuation(ch: str) -> bool:
    if len(ch) != 1:
        return False
    if ch in COMMON_PUNCTUATION:
        return True
    return unicodedata.category(ch).startswith("P")


def _minimum_base_vocab_size(extra_tokens: Sequence[str] = ()) -> int:
    return len(_unique(list(SPECIAL_TOKENS) + list(extra_tokens) + list(ASCII_FALLBACK) + list(COMMON_PUNCTUATION)))


def load_hf_vocab(path: str | Path) -> list[str]:
    """Load tokens from a HF `vocab.txt` or WordPiece `tokenizer.json`.

    BPE/SentencePiece `tokenizer.json` files also contain vocab maps, but their
    merge/model rules are different. This tokenizer expects WordPiece tokens.
    """

    path = Path(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        vocab = payload.get("model", {}).get("vocab") or payload.get("vocab")
        if not isinstance(vocab, dict):
            raise ValueError(f"{path} does not contain a HF vocab map")
        return [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]

    return [
        line.rstrip("\n")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prune_hf_vocab(
    source_tokens: Sequence[str],
    *,
    max_vocab_size: int = 4096,
    max_chinese_chars: int = 1000,
    max_english_words: int = 2000,
    max_english_pieces: int = 1000,
    max_cjk_words: int = 0,
    extra_tokens: Sequence[str] = (),
) -> list[str]:
    """Build a compact vocabulary from a full HF WordPiece vocabulary.

    The source vocab order is treated as the priority order. BERT vocabularies
    generally put more common tokens earlier, so clipping by category gives a
    useful small model vocab without hand-maintained word lists.
    """

    min_base_size = _minimum_base_vocab_size(extra_tokens)
    if max_vocab_size < min_base_size:
        raise ValueError(f"max_vocab_size must be at least {min_base_size} for specials, ASCII, and punctuation")

    source_tokens = _unique(source_tokens)
    source_set = set(source_tokens)
    tokens: list[str] = list(SPECIAL_TOKENS)

    def add(token: str) -> bool:
        if len(tokens) >= max_vocab_size:
            return False
        if token in tokens or not token:
            return False
        tokens.append(token)
        return True

    for token in extra_tokens:
        add(token)

    # Keep exact text reconstruction practical. Spaces are custom specials;
    # punctuation and ASCII fallback characters should exist even after pruning.
    for token in ASCII_FALLBACK:
        add(token)
    for token in COMMON_PUNCTUATION:
        add(token)

    chinese_count = 0
    for token in source_tokens:
        if chinese_count >= max_chinese_chars or len(tokens) >= max_vocab_size:
            break
        if _is_single_cjk(token):
            chinese_count += int(add(token))

    english_word_count = 0
    for token in source_tokens:
        if english_word_count >= max_english_words or len(tokens) >= max_vocab_size:
            break
        if _is_whole_ascii_word(token):
            english_word_count += int(add(token.lower()))

    english_piece_count = 0
    for token in source_tokens:
        if english_piece_count >= max_english_pieces or len(tokens) >= max_vocab_size:
            break
        if _is_ascii_continuation(token):
            english_piece_count += int(add(token.lower()))

    cjk_word_count = 0
    if max_cjk_words > 0:
        for token in source_tokens:
            if cjk_word_count >= max_cjk_words or len(tokens) >= max_vocab_size:
                break
            if len(token) > 1 and _contains_cjk(token) and _is_supported_source_token(token):
                cjk_word_count += int(add(token))

    # Fill the remaining budget with supported source tokens in HF order.
    for token in source_tokens:
        if len(tokens) >= max_vocab_size:
            break
        if not _is_supported_source_token(token):
            continue
        if token in source_set and token.startswith("##"):
            add(token.lower())
        elif _contains_cjk(token) or _is_ascii_wordpiece(token) or len(token) == 1:
            add(token.lower() if token.isascii() else token)

    return tokens[:max_vocab_size]


def prune_hf_vocab_for_corpus(
    source_tokens: Sequence[str],
    texts: Iterable[str],
    *,
    max_vocab_size: int = 4096,
    max_chinese_chars: int = 1000,
    max_english_words: int = 2000,
    max_english_pieces: int = 1000,
    max_cjk_words: int = 0,
    lowercase: bool = True,
    normalize: str = "NFC",
    extra_tokens: Sequence[str] = (),
) -> list[str]:
    """Build a compact HF-derived vocabulary prioritized by corpus frequency."""

    min_base_size = _minimum_base_vocab_size(extra_tokens)
    if max_vocab_size < min_base_size:
        raise ValueError(f"max_vocab_size must be at least {min_base_size} for specials, ASCII, and punctuation")

    source_tokens = _unique(source_tokens)
    source_set = set(source_tokens)
    source_lower_set = {token.lower() for token in source_tokens if token.isascii()}
    tokens: list[str] = list(SPECIAL_TOKENS)
    chinese_counts: dict[str, int] = {}
    english_counts: dict[str, int] = {}

    for text in texts:
        if normalize:
            text = unicodedata.normalize(normalize, text)
        if lowercase:
            text = text.lower()
        for ch in text:
            if _is_cjk_char(ch):
                chinese_counts[ch] = chinese_counts.get(ch, 0) + 1
        for match in WORD_RE.finditer(text):
            word = match.group(0)
            english_counts[word] = english_counts.get(word, 0) + 1

    def add(token: str) -> bool:
        if len(tokens) >= max_vocab_size:
            return False
        if token in tokens or not token:
            return False
        tokens.append(token)
        return True

    for token in extra_tokens:
        add(token)
    for token in ASCII_FALLBACK:
        add(token)
    for token in COMMON_PUNCTUATION:
        add(token)

    chinese_added = 0
    for token, _ in sorted(chinese_counts.items(), key=lambda item: item[1], reverse=True):
        if chinese_added >= max_chinese_chars or len(tokens) >= max_vocab_size:
            break
        if token in source_set:
            chinese_added += int(add(token))

    english_added = 0
    for token, _ in sorted(english_counts.items(), key=lambda item: item[1], reverse=True):
        if english_added >= max_english_words or len(tokens) >= max_vocab_size:
            break
        source_token = token.lower() if lowercase else token
        if source_token in source_set or source_token in source_lower_set:
            english_added += int(add(source_token))

    english_piece_count = 0
    for token in source_tokens:
        if english_piece_count >= max_english_pieces or len(tokens) >= max_vocab_size:
            break
        if _is_ascii_continuation(token):
            english_piece_count += int(add(token.lower() if lowercase else token))

    cjk_word_count = 0
    if max_cjk_words > 0:
        for token in source_tokens:
            if cjk_word_count >= max_cjk_words or len(tokens) >= max_vocab_size:
                break
            if len(token) > 1 and _contains_cjk(token) and _is_supported_source_token(token):
                cjk_word_count += int(add(token))

    for token in source_tokens:
        if len(tokens) >= max_vocab_size:
            break
        if not _is_supported_source_token(token):
            continue
        if token.startswith("##"):
            add(token.lower() if lowercase else token)
        elif _contains_cjk(token) or _is_ascii_wordpiece(token) or len(token) == 1:
            add(token.lower() if lowercase and token.isascii() else token)

    return tokens[:max_vocab_size]


class HFWordPieceTokenizer:
    """Small WordPiece tokenizer using a pruned Hugging Face vocabulary."""

    def __init__(
        self,
        tokens: Sequence[str],
        *,
        lowercase: bool = True,
        normalize: str = "NFC",
        max_input_chars_per_word: int = 100,
    ):
        tokens = _unique(tokens)
        missing = [token for token in SPECIAL_TOKENS if token not in tokens]
        if missing:
            raise ValueError(f"missing special tokens: {missing}")

        self.config = TokenizerConfig(
            lowercase=lowercase,
            normalize=normalize,
            max_input_chars_per_word=max_input_chars_per_word,
        )
        self.id_to_token = list(tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}

        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]
        self.bos_id = self.token_to_id[BOS_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @classmethod
    def from_vocab(
        cls,
        source_tokens: Sequence[str],
        *,
        max_vocab_size: int = 4096,
        max_chinese_chars: int = 1000,
        max_english_words: int = 2000,
        max_english_pieces: int = 1000,
        max_cjk_words: int = 0,
        lowercase: bool = True,
        normalize: str = "NFC",
    ) -> "HFWordPieceTokenizer":
        tokens = prune_hf_vocab(
            source_tokens,
            max_vocab_size=max_vocab_size,
            max_chinese_chars=max_chinese_chars,
            max_english_words=max_english_words,
            max_english_pieces=max_english_pieces,
            max_cjk_words=max_cjk_words,
        )
        return cls(tokens, lowercase=lowercase, normalize=normalize)

    @classmethod
    def from_vocab_file(
        cls,
        path: str | Path,
        *,
        max_vocab_size: int = 4096,
        max_chinese_chars: int = 1000,
        max_english_words: int = 2000,
        max_english_pieces: int = 1000,
        max_cjk_words: int = 0,
        lowercase: bool = True,
        normalize: str = "NFC",
    ) -> "HFWordPieceTokenizer":
        return cls.from_vocab(
            load_hf_vocab(path),
            max_vocab_size=max_vocab_size,
            max_chinese_chars=max_chinese_chars,
            max_english_words=max_english_words,
            max_english_pieces=max_english_pieces,
            max_cjk_words=max_cjk_words,
            lowercase=lowercase,
            normalize=normalize,
        )

    @classmethod
    def from_vocab_file_and_corpus(
        cls,
        vocab_path: str | Path,
        corpus_texts: Iterable[str],
        *,
        max_vocab_size: int = 4096,
        max_chinese_chars: int = 1000,
        max_english_words: int = 2000,
        max_english_pieces: int = 1000,
        max_cjk_words: int = 0,
        lowercase: bool = True,
        normalize: str = "NFC",
    ) -> "HFWordPieceTokenizer":
        tokens = prune_hf_vocab_for_corpus(
            load_hf_vocab(vocab_path),
            corpus_texts,
            max_vocab_size=max_vocab_size,
            max_chinese_chars=max_chinese_chars,
            max_english_words=max_english_words,
            max_english_pieces=max_english_pieces,
            max_cjk_words=max_cjk_words,
            lowercase=lowercase,
            normalize=normalize,
        )
        return cls(tokens, lowercase=lowercase, normalize=normalize)

    @classmethod
    def load(cls, path: str | Path) -> "HFWordPieceTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            payload["tokens"],
            lowercase=payload.get("lowercase", True),
            normalize=payload.get("normalize", "NFC"),
            max_input_chars_per_word=payload.get("max_input_chars_per_word", 100),
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "type": "HFWordPieceTokenizer",
            "lowercase": self.config.lowercase,
            "normalize": self.config.normalize,
            "max_input_chars_per_word": self.config.max_input_chars_per_word,
            "tokens": self.id_to_token,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: int | None = None,
        pad_to_max: bool = False,
        truncate: bool = True,
        return_np: bool = True,
    ) -> np.ndarray | list[int]:
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.bos_id)

        token_ids.extend(self.token_to_id.get(token, self.unk_id) for token in self.tokenize(text))

        if add_eos:
            token_ids.append(self.eos_id)

        if max_len is not None:
            if len(token_ids) > max_len:
                if not truncate:
                    raise ValueError(f"encoded length {len(token_ids)} exceeds max_len={max_len}")
                token_ids = token_ids[:max_len]
                if add_eos and max_len > 0:
                    token_ids[-1] = self.eos_id
            if pad_to_max:
                token_ids = token_ids + [self.pad_id] * (max_len - len(token_ids))

        if return_np:
            return np.array(token_ids, dtype=np.int32)
        return token_ids

    def batch_encode(
        self,
        texts: Sequence[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: int,
    ) -> np.ndarray:
        return np.stack(
            [
                self.encode(
                    text,
                    add_bos=add_bos,
                    add_eos=add_eos,
                    max_len=max_len,
                    pad_to_max=True,
                    return_np=True,
                )
                for text in texts
            ]
        )

    def decode(self, token_ids: Sequence[int], *, skip_special: bool = True) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            token = self.id_to_token[int(token_id)]
            if skip_special and token in CONTROL_TOKENS:
                continue
            if token == SPACE_TOKEN:
                pieces.append(" ")
            elif token == NEWLINE_TOKEN:
                pieces.append("\n")
            elif token == TAB_TOKEN:
                pieces.append("\t")
            elif token.startswith("##"):
                pieces.append(token[2:])
            else:
                pieces.append(token)
        return "".join(pieces)

    def tokenize(self, text: str) -> list[str]:
        text = self._normalize_text(text)
        tokens: list[str] = []
        i = 0
        while i < len(text):
            ch = text[i]

            if ch == " ":
                tokens.append(SPACE_TOKEN)
                i += 1
                continue
            if ch == "\n":
                tokens.append(NEWLINE_TOKEN)
                i += 1
                continue
            if ch == "\t":
                tokens.append(TAB_TOKEN)
                i += 1
                continue
            if ch.isspace():
                tokens.append(SPACE_TOKEN)
                i += 1
                continue
            if _is_cjk_char(ch):
                tokens.append(ch if ch in self.token_to_id else UNK_TOKEN)
                i += 1
                continue

            match = WORD_RE.match(text, i)
            if match is not None:
                tokens.extend(self._wordpiece(match.group(0)))
                i = match.end()
                continue

            tokens.append(ch if ch in self.token_to_id else UNK_TOKEN)
            i += 1

        return tokens

    def _wordpiece(self, token: str) -> list[str]:
        if len(token) > self.config.max_input_chars_per_word:
            return self._char_fallback(token)
        if token in self.token_to_id:
            return [token]

        sub_tokens: list[str] = []
        start = 0
        while start < len(token):
            end = len(token)
            current = None
            while start < end:
                piece = token[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in self.token_to_id:
                    current = piece
                    break
                end -= 1

            if current is None:
                return self._char_fallback(token)

            sub_tokens.append(current)
            start = end

        return sub_tokens

    def _char_fallback(self, token: str) -> list[str]:
        out: list[str] = []
        for ch in token:
            if ch in self.token_to_id:
                out.append(ch)
            elif _is_cjk_char(ch) and ch in self.token_to_id:
                out.append(ch)
            else:
                out.append(UNK_TOKEN)
        return out

    def _normalize_text(self, text: str) -> str:
        if self.config.normalize:
            text = unicodedata.normalize(self.config.normalize, text)
        if self.config.lowercase:
            text = text.lower()
        return text


def build_tokenizer_from_hf_vocab(
    vocab_path: str | Path,
    *,
    max_vocab_size: int = 4096,
    max_chinese_chars: int = 1000,
    max_english_words: int = 2000,
    max_english_pieces: int = 1000,
    lowercase: bool = True,
) -> HFWordPieceTokenizer:
    return HFWordPieceTokenizer.from_vocab_file(
        vocab_path,
        max_vocab_size=max_vocab_size,
        max_chinese_chars=max_chinese_chars,
        max_english_words=max_english_words,
        max_english_pieces=max_english_pieces,
        lowercase=lowercase,
    )


# Backward-compatible alias for earlier local experiments.
HybridTokenizer = HFWordPieceTokenizer


def _demo_source_vocab() -> list[str]:
    return [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "!",
        ".",
        "，",
        "。",
        "hello",
        "world",
        "transform",
        "##er",
        "simple",
        "text",
        "a",
        "i",
        "n",
        "r",
        "s",
        "t",
        "1",
        "2",
        "3",
        "人",
        "工",
        "智",
        "能",
        "正",
        "在",
        "学",
        "习",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small tokenizer from a HF WordPiece vocab.")
    parser.add_argument("--source-vocab", type=Path, help="HF vocab.txt or WordPiece tokenizer.json")
    parser.add_argument("--save", type=Path, help="Where to save the compact tokenizer JSON")
    parser.add_argument("--max-vocab-size", type=int, default=4096)
    parser.add_argument("--max-chinese-chars", type=int, default=1000)
    parser.add_argument("--max-english-words", type=int, default=2000)
    parser.add_argument("--max-english-pieces", type=int, default=1000)
    parser.add_argument("--no-lowercase", action="store_true")
    args = parser.parse_args()

    if args.source_vocab is None:
        tokenizer = HFWordPieceTokenizer.from_vocab(_demo_source_vocab(), max_vocab_size=256)
    else:
        tokenizer = HFWordPieceTokenizer.from_vocab_file(
            args.source_vocab,
            max_vocab_size=args.max_vocab_size,
            max_chinese_chars=args.max_chinese_chars,
            max_english_words=args.max_english_words,
            max_english_pieces=args.max_english_pieces,
            lowercase=not args.no_lowercase,
        )

    if args.save is not None:
        tokenizer.save(args.save)

    sample = "Hello transformer，人工智能正在学习 simple text."
    ids = tokenizer.encode(sample, add_bos=True, add_eos=True, max_len=64, pad_to_max=True)
    print("vocab_size:", tokenizer.vocab_size)
    print("ids:", ids.tolist())
    print("tokens:", tokenizer.tokenize(sample))
    print("decoded:", tokenizer.decode(ids))


if __name__ == "__main__":
    main()
