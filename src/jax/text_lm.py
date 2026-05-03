"""Train the JAX causal text language model.

This file is intentionally only the executable training entry point. The
corpus schema/token cache lives in corpus.py, the model architecture lives in
lm_model.py, and optimizer/checkpoint logic lives in trainer.py.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

try:
    from .config_loader import load_cfg, parse_bootstrap_args, setup_jax_environment
except ImportError:
    from config_loader import load_cfg, parse_bootstrap_args, setup_jax_environment


_BOOTSTRAP_ARGS = parse_bootstrap_args(sys.argv[1:]) if __name__ == "__main__" else argparse.Namespace(config=None)
CFG, TextLMConfig, CONFIG_SOURCE = load_cfg(_BOOTSTRAP_ARGS.config, __package__, "text_lm")
setup_jax_environment(CFG.jax_platforms)

import jax

try:
    from .corpus import encode_or_load_training_data, load_or_build_tokenizer, random_batch, read_corpus_records
    from .lm_model import build_meta
    from .trainer import (
        estimate_losses,
        generate_text,
        init_or_load_state,
        normalize_loss_objective,
        save_checkpoint,
        seed_everything,
        train_step,
        write_run_config,
    )
except ImportError:
    from corpus import encode_or_load_training_data, load_or_build_tokenizer, random_batch, read_corpus_records
    from lm_model import build_meta
    from trainer import (
        estimate_losses,
        generate_text,
        init_or_load_state,
        normalize_loss_objective,
        save_checkpoint,
        seed_everything,
        train_step,
        write_run_config,
    )


def resolve_config(cfg: TextLMConfig) -> TextLMConfig:
    return cfg


def build_model_meta(cfg, vocab_size: int) -> dict:
    return build_meta(
        vocab_size=vocab_size,
        max_seqlen=cfg.block_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        n_layers=cfg.n_layers,
        scale_token_embeddings=cfg.scale_token_embeddings,
        final_norm=cfg.final_norm,
        hyperconnection_streams=cfg.hyperconnection_streams,
        hyperconnection_mode=getattr(cfg, "hyperconnection_mode", None),
        hyperconnection_dynamic=getattr(cfg, "hyperconnection_dynamic", None),
        hyperconnection_sinkhorn_iters=cfg.hyperconnection_sinkhorn_iters,
        attention_implementation=getattr(cfg, "attention_implementation", "cudnn"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the JAX causal text language model.")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Python module name or .py file that defines CFG. Example: --config src/jax/init.py",
    )
    return parser.parse_args()


def main(cfg: TextLMConfig = CFG) -> None:
    cfg = resolve_config(cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("config:", CONFIG_SOURCE)
    print("JAX version:", jax.__version__)
    print("devices:", jax.devices())

    records, text, record_count, corpus_format = read_corpus_records(cfg)
    print(f"loaded corpus: {cfg.corpus} format={corpus_format} records={record_count} chars={len(text)}")

    tokenizer = load_or_build_tokenizer(cfg, text)
    training_data = encode_or_load_training_data(cfg, tokenizer, text, records)
    meta = build_model_meta(cfg, tokenizer.vocab_size)
    write_run_config(cfg.out_dir / "run_config.json", cfg, meta, tokenizer, len(training_data.tokens), CONFIG_SOURCE)
    print("model meta:", meta)
    if (
        getattr(cfg, "loss_objective", "ce").lower() == "dft"
        and cfg.init_checkpoint is None
        and not cfg.checkpoint.exists()
    ):
        print(
            "warning: loss_objective=dft is intended for fine-tuning from a pretrained checkpoint; "
            "from-scratch training usually should start with ce"
        )

    if cfg.dry_run:
        x, y = random_batch(training_data, min(cfg.batch_size, 2), cfg.block_size, np.random.default_rng(cfg.seed))
        print("dry run batch:", x.shape, y.shape)
        print("dry run ignored targets:", int(np.sum(y < 0)))
        print("sample decoded:", tokenizer.decode(x[0][: min(80, len(x[0]))]))
        return

    key = seed_everything(cfg.seed)
    state, configs, start_step, key = init_or_load_state(
        cfg.checkpoint,
        meta,
        cfg,
        key,
        cfg.init_checkpoint,
    )
    rng = np.random.default_rng(cfg.seed + start_step)
    start_time = time.time()
    ema_loss: float | None = None
    ema_ce_loss: float | None = None
    loss_objective = normalize_loss_objective(getattr(cfg, "loss_objective", "ce"))
    label_smoothing = float(getattr(cfg, "label_smoothing", 0.0))
    log_smoothing_metrics = loss_objective in {"ce", "ce_dft"} and label_smoothing > 0.0
    log_dft_metrics = loss_objective in {"dft", "ce_dft"}

    for step in range(start_step + 1, cfg.steps + 1):
        x, y = random_batch(training_data, cfg.batch_size, cfg.block_size, rng)
        key, subkey = jax.random.split(key)
        state, metrics = train_step(x, y, state, configs, subkey, step)

        if step == 1 or step % cfg.log_every == 0:
            loss = metrics["loss"]
            loss.block_until_ready()
            loss_value = float(loss)
            ce_loss_value = float(metrics["ce_loss"])
            ema_loss = loss_value if ema_loss is None else 0.9 * ema_loss + 0.1 * loss_value
            ema_ce_loss = ce_loss_value if ema_ce_loss is None else 0.9 * ema_ce_loss + 0.1 * ce_loss_value
            elapsed = time.time() - start_time
            log_line = f"step={step} loss={loss_value:.4f} loss_ema={ema_loss:.4f}"
            if log_smoothing_metrics or log_dft_metrics:
                log_line += (
                    f" ce_loss={ce_loss_value:.4f} ce_ema={ema_ce_loss:.4f}"
                )
            if log_smoothing_metrics:
                log_line += (
                    f" smooth_ce={float(metrics['smoothed_ce_loss']):.4f}"
                    f" label_smoothing={float(metrics['label_smoothing']):.3f}"
                )
            if log_dft_metrics:
                log_line += (
                    f" dft_loss={float(metrics['dft_loss']):.4f}"
                    f" dft_alpha={float(metrics['dft_alpha']):.3f}"
                    f" p_gold={float(metrics['mean_gold_prob']):.4f}"
                )
            if log_smoothing_metrics or log_dft_metrics:
                log_line += (
                    f" valid_tokens={int(float(metrics['valid_tokens']))}"
                )
            print(f"{log_line} elapsed={elapsed:.1f}s")

        if step % cfg.save_every == 0 or step == cfg.steps:
            save_checkpoint(cfg.checkpoint, state, meta, step, cfg)
            print(f"saved checkpoint: {cfg.checkpoint} step={step}")

    if cfg.eval_batches > 0:
        key, subkey = jax.random.split(key)
        eval_metrics = estimate_losses(training_data, state, configs, subkey, cfg, step=cfg.steps)
        eval_line = f"eval_loss={eval_metrics['loss']:.4f}"
        if log_smoothing_metrics or log_dft_metrics:
            eval_line += (
                f" eval_ce_loss={eval_metrics['ce_loss']:.4f}"
            )
        if log_smoothing_metrics:
            eval_line += (
                f" eval_smooth_ce={eval_metrics['smoothed_ce_loss']:.4f}"
                f" eval_label_smoothing={eval_metrics['label_smoothing']:.3f}"
            )
        if log_dft_metrics:
            eval_line += (
                f" eval_dft_loss={eval_metrics['dft_loss']:.4f}"
                f" eval_dft_alpha={eval_metrics['dft_alpha']:.3f}"
                f" eval_p_gold={eval_metrics['mean_gold_prob']:.4f}"
            )
        print(eval_line)

    if cfg.sample_prompt:
        print(
            generate_text(
                cfg.sample_prompt,
                tokenizer,
                state,
                configs,
                block_size=cfg.block_size,
                max_new_tokens=cfg.sample_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                seed=cfg.seed + 123,
            )
        )


if __name__ == "__main__":
    parse_args()
    main()
