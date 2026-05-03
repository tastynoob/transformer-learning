"""Default configuration for the JAX text language model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextLMConfig:
    # =========================================================================
    # Mutable hot knobs.
    # 最常改的参数放在最上面。切换 corpus/loss_mode/loss_objective/block_size 时建议同步换
    # out_dir/token_cache/checkpoint，避免不同训练口径混到同一个 run。
    # =========================================================================
    # Use src/jax/prepare_dialogue_logic_mix_corpus.py to build this JSONL corpus.
    corpus: Path = Path("data/corpus/dialogue_logic_mix_1024.tl.jsonl")
    # record: use each normalized segment's train flag; assistant: train only assistant tokens; all: pretrain-style LM.
    loss_mode: str = "record"
    # ce: standard SFT/NLL; dft: pure Dynamic Fine-Tuning; ce_dft: stable CE+DFT mixture.
    loss_objective: str = "ce_dft"
    dft_alpha: float = 0.2
    dft_start_step: int = 0
    dft_warmup_steps: int = 2000
    label_smoothing: float = 0.01
    block_size: int = 1024
    steps: int = 80000
    batch_size: int = 16
    lr: float = 5e-5
    attention_implementation: str = "cudnn"

    out_dir: Path = Path("runs/jax_text_lm_dialogue_logic_mix_1024")
    tokenizer_json: Path = Path("runs/jax_text_lm_dialogue_logic_mix_1024/tokenizer.json")
    token_cache: Path = Path("runs/jax_text_lm_dialogue_logic_mix_1024/tokens.npy")
    checkpoint: Path = Path("runs/jax_text_lm_dialogue_logic_mix_1024/checkpoint.pkl")

    init_checkpoint: Path | None = None # Path("runs/mode_en/checkpoint.pkl")
    init_tokenizer_json: Path | None = None # Path("runs/mode_en/tokenizer.json")

    seed: int = 0
    jax_platforms: str | None = "gpu"

    # =========================================================================
    # Mutable corpus/cache details.
    # 较少改；用于非标准输入、debug 子集、强制重建 tokenizer/cache。
    # =========================================================================
    corpus_format: str = "auto"  # auto, text, jsonl, json
    corpus_text_field: str = "text"
    corpus_joiner: str = "\n"
    max_chars: int | None = None
    # True: each sampled training window stays inside one record. Long records
    # are cropped; short records are right-padded with ignored targets.
    record_aware_batches: bool = True
    rebuild_tokenizer: bool = False
    retokenize: bool = False

    # =========================================================================
    # Mutable optimizer details.
    # lr 放在 hot knobs；这里是一般不频繁改的优化器细节。
    # =========================================================================
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    lr_warmup_steps: int = 2000
    min_lr_ratio: float = 0.1

    # =========================================================================
    # Mutable logging, checkpointing, validation, and preview.
    # 只影响观测、保存频率、dry-run 和训练结束后的 sample 预览。
    # =========================================================================
    save_every: int = 100
    log_every: int = 10
    eval_batches: int = 10
    dry_run: bool = False
    sample_prompt: str = ""
    sample_tokens: int = 80
    temperature: float = 0.9
    top_k: int = 40

    # =========================================================================
    # Training-invariant tokenizer contract.
    # 这些决定 token id；继续训练已有 checkpoint 时不能随便改。
    # =========================================================================
    # Download a Hugging Face WordPiece vocab.txt here before the first run.
    source_vocab: Path = Path("data/hf/bert-base-multilingual-cased/vocab.txt")
    vocab_size: int = 8192
    max_chinese_chars: int = 2000
    max_english_words: int = 2000
    max_english_pieces: int = 1000
    max_cjk_words: int = 1000
    lowercase: bool = True

    # =========================================================================
    # Training-invariant model architecture.
    # 这些影响 checkpoint 结构或 forward 语义；改了通常要新建模型。
    # =========================================================================
    d_model: int = 512
    n_heads: int = 4
    d_ff: int = 2048
    n_layers: int = 8

    # Fixed sinusoidal position encodings have O(1) amplitude, so token
    # embeddings need the original Transformer sqrt(d_model) scale.
    scale_token_embeddings: bool = True
    final_norm: bool = True

    # Manifold-Constrained Hyper-Connections. Set streams to 1 to disable.
    hyperconnection_streams: int = 4
    hyperconnection_mode: str | None = None  # None -> sublayer when streams > 1.
    hyperconnection_dynamic: bool | None = None  # None -> enabled for sublayer mHC.
    hyperconnection_sinkhorn_iters: int = 8


CFG = TextLMConfig()
