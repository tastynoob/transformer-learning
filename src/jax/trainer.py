"""Training, optimizer state, checkpoint, and sampling helpers."""

from __future__ import annotations

from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import pickle
import random
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

try:
    from .configs import StaticConfigs
    from .corpus import IGNORE_INDEX, TrainingData, random_batch
    from .layer import adamOpt_init, adamWOpt_init, dataType
    from .lm_model import build_model, lm_apply
    from .tokenizer import HFWordPieceTokenizer
except ImportError:
    from configs import StaticConfigs
    from corpus import IGNORE_INDEX, TrainingData, random_batch
    from layer import adamOpt_init, adamWOpt_init, dataType
    from lm_model import build_model, lm_apply
    from tokenizer import HFWordPieceTokenizer


MODEL_COMPAT_META_KEYS = (
    "kind",
    "vocab_size",
    "d_model",
    "d_ff",
    "d_k",
    "d_v",
    "n_heads",
    "n_layers",
    "scale_token_embeddings",
    "final_norm",
    "hyperconnection_streams",
    "hyperconnection_mode",
    "hyperconnection_dynamic",
)


def seed_everything(seed: int) -> jax.Array:
    random.seed(seed)
    np.random.seed(seed)
    return jax.random.PRNGKey(seed)


def batch_lm_apply(
    input_batch,
    model,
    n_heads: int,
    drop_prob: float,
    sinkhorn_iters: int,
    scale_token_embeddings: bool,
    attention_implementation: str,
    key,
):
    batch_size = input_batch.shape[0]
    keys = jax.random.split(key, batch_size)

    def apply_one(input_ids, subkey):
        return lm_apply(
            input_ids,
            model,
            n_heads,
            drop_prob,
            subkey,
            sinkhorn_iters,
            scale_token_embeddings,
            attention_implementation,
        )

    return jax.vmap(apply_one)(input_batch, keys)


def normalize_loss_objective(loss_objective: str) -> str:
    objective = loss_objective.lower()
    if objective in {"ce", "sft", "nll", "cross_entropy"}:
        return "ce"
    if objective in {"ce_dft", "dft_mix", "mixed_dft"}:
        return "ce_dft"
    if objective == "dft":
        return "dft"
    raise ValueError("loss_objective must be one of: ce, sft, nll, cross_entropy, dft, ce_dft")


def uses_dft_objective(loss_objective: str) -> bool:
    return normalize_loss_objective(loss_objective) in {"dft", "ce_dft"}


def uses_smoothed_ce(loss_objective: str, label_smoothing: float) -> bool:
    return normalize_loss_objective(loss_objective) in {"ce", "ce_dft"} and float(label_smoothing) > 0.0


def dft_alpha_for_step(step, max_alpha: float, start_step: int, warmup_steps: int):
    step_f = jnp.asarray(step, dtype=jnp.float32)
    max_alpha_f = jnp.asarray(max_alpha, dtype=jnp.float32)
    start_f = jnp.asarray(start_step, dtype=jnp.float32)
    if warmup_steps <= 0:
        return jnp.where(step_f >= start_f, max_alpha_f, 0.0)
    warmup_f = jnp.asarray(warmup_steps, dtype=jnp.float32)
    progress = jnp.clip((step_f - start_f) / warmup_f, 0.0, 1.0)
    return max_alpha_f * progress


def masked_lm_losses(
    target_ids,
    logits,
    loss_objective: str = "ce",
    ignore_index: int = IGNORE_INDEX,
    dft_alpha=1.0,
    label_smoothing: float = 0.0,
):
    objective = normalize_loss_objective(loss_objective)
    apply_smoothing = uses_smoothed_ce(objective, label_smoothing)
    apply_dft = objective in {"dft", "ce_dft"}

    target_ids = jnp.asarray(target_ids)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    valid = target_ids != ignore_index
    valid_float = valid.astype(logits.dtype)
    safe_targets = jnp.where(valid, target_ids, 0)
    gold_log_probs = jnp.take_along_axis(log_probs, safe_targets[..., None], axis=-1).squeeze(-1)
    token_losses = -gold_log_probs
    valid_count = jnp.sum(valid_float)
    denom = jnp.maximum(valid_count, 1.0)
    ce_loss = jnp.sum(jnp.where(valid, token_losses, 0.0)) / denom

    if apply_smoothing:
        smoothing_value = max(0.0, min(float(label_smoothing), 1.0))
        smoothing = jnp.asarray(smoothing_value, dtype=jnp.float32)
        uniform_token_losses = -jnp.mean(log_probs, axis=-1)
        uniform_loss = jnp.sum(jnp.where(valid, uniform_token_losses, 0.0)) / denom
        smoothed_ce_loss = (1.0 - smoothing) * ce_loss + smoothing * uniform_loss
    else:
        smoothing = jnp.asarray(0.0, dtype=jnp.float32)
        smoothed_ce_loss = ce_loss

    if apply_dft:
        gold_probs = jnp.exp(gold_log_probs)
        mean_gold_prob = jnp.sum(jnp.where(valid, gold_probs, 0.0)) / denom
        dft_weights = jax.lax.stop_gradient(gold_probs)
        dft_loss = jnp.sum(jnp.where(valid, dft_weights * token_losses, 0.0)) / denom
    else:
        mean_gold_prob = jnp.asarray(0.0, dtype=jnp.float32)
        dft_loss = jnp.asarray(0.0, dtype=jnp.float32)

    if objective == "ce":
        return (
            smoothed_ce_loss,
            ce_loss,
            smoothed_ce_loss,
            dft_loss,
            mean_gold_prob,
            valid_count,
            jnp.asarray(0.0, dtype=jnp.float32),
            smoothing,
        )
    if objective == "dft":
        return (
            dft_loss,
            ce_loss,
            smoothed_ce_loss,
            dft_loss,
            mean_gold_prob,
            valid_count,
            jnp.asarray(1.0, dtype=jnp.float32),
            smoothing,
        )

    alpha = jnp.asarray(dft_alpha, dtype=jnp.float32)
    mixed_loss = (1.0 - alpha) * smoothed_ce_loss + alpha * dft_loss
    return (
        mixed_loss,
        ce_loss,
        smoothed_ce_loss,
        dft_loss,
        mean_gold_prob,
        valid_count,
        alpha,
        smoothing,
    )


def tree_global_norm(tree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    sq_sums = [jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves]
    return jnp.sqrt(jnp.sum(jnp.asarray(sq_sums, dtype=jnp.float32)))


def clip_grads_by_global_norm(grads, max_norm: float):
    if max_norm <= 0:
        return grads
    norm = tree_global_norm(grads)
    scale = jnp.minimum(1.0, jnp.asarray(max_norm, dtype=jnp.float32) / (norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale.astype(g.dtype), grads)


@partial(jax.jit, static_argnames=("configs",))
def train_step(input_batch, target_batch, state, configs, key, step):
    model = state["model"]
    opt_state = state["opt_state"]
    objective = configs["loss_objective"]
    dft_alpha = (
        dft_alpha_for_step(
            step,
            configs["dft_alpha"],
            configs["dft_start_step"],
            configs["dft_warmup_steps"],
        )
        if objective == "ce_dft"
        else jnp.asarray(1.0 if objective == "dft" else 0.0, dtype=jnp.float32)
    )

    def compute_loss(params):
        logits = batch_lm_apply(
            input_batch,
            params,
            configs["n_heads"],
            configs["drop_prob"],
            configs["hyperconnection_sinkhorn_iters"],
            configs["scale_token_embeddings"],
            configs["attention_implementation"],
            key,
        )
        (
            objective_loss,
            ce_loss,
            smoothed_ce_loss,
            dft_loss,
            mean_gold_prob,
            valid_count,
            used_dft_alpha,
            used_label_smoothing,
        ) = masked_lm_losses(
            target_batch,
            logits,
            configs["loss_objective"],
            dft_alpha=dft_alpha,
            label_smoothing=configs["label_smoothing"],
        )
        return objective_loss, {
            "ce_loss": ce_loss,
            "smoothed_ce_loss": smoothed_ce_loss,
            "dft_loss": dft_loss,
            "mean_gold_prob": mean_gold_prob,
            "valid_tokens": valid_count,
            "dft_alpha": used_dft_alpha,
            "label_smoothing": used_label_smoothing,
        }

    (avg_loss, aux_metrics), avg_grads = jax.value_and_grad(compute_loss, has_aux=True)(model)
    avg_grads = clip_grads_by_global_norm(avg_grads, configs["grad_clip_norm"])

    updates, new_opt_state = configs["opt_fn"](avg_grads, opt_state, configs, model)
    new_model = jax.tree_util.tree_map(lambda p, u: p - u, model, updates)
    return {
        "model": new_model,
        "opt_state": new_opt_state,
    }, {
        "loss": avg_loss,
        **aux_metrics,
    }


@partial(jax.jit, static_argnames=("configs",))
def eval_step(input_batch, target_batch, state, configs, key, step):
    model = state["model"]
    objective = configs["loss_objective"]
    dft_alpha = (
        dft_alpha_for_step(
            step,
            configs["dft_alpha"],
            configs["dft_start_step"],
            configs["dft_warmup_steps"],
        )
        if objective == "ce_dft"
        else jnp.asarray(1.0 if objective == "dft" else 0.0, dtype=jnp.float32)
    )
    logits = batch_lm_apply(
        input_batch,
        model,
        configs["n_heads"],
        -1.0,
        configs["hyperconnection_sinkhorn_iters"],
        configs["scale_token_embeddings"],
        configs["attention_implementation"],
        key,
    )
    (
        objective_loss,
        ce_loss,
        smoothed_ce_loss,
        dft_loss,
        mean_gold_prob,
        valid_count,
        used_dft_alpha,
        used_label_smoothing,
    ) = masked_lm_losses(
        target_batch,
        logits,
        configs["loss_objective"],
        dft_alpha=dft_alpha,
        label_smoothing=configs["label_smoothing"],
    )
    return {
        "loss": objective_loss,
        "ce_loss": ce_loss,
        "smoothed_ce_loss": smoothed_ce_loss,
        "dft_loss": dft_loss,
        "mean_gold_prob": mean_gold_prob,
        "valid_tokens": valid_count,
        "dft_alpha": used_dft_alpha,
        "label_smoothing": used_label_smoothing,
    }


def estimate_losses(data: TrainingData, state, configs, key, cfg, step: int | None = None) -> dict[str, float]:
    losses: list[float] = []
    ce_losses: list[float] = []
    smoothed_ce_losses: list[float] = []
    dft_losses: list[float] = []
    mean_gold_probs: list[float] = []
    valid_tokens: list[float] = []
    dft_alphas: list[float] = []
    label_smoothings: list[float] = []
    rng = np.random.default_rng(cfg.seed + 999)
    eval_step_index = cfg.steps if step is None else step
    for _ in range(cfg.eval_batches):
        x, y = random_batch(data, cfg.batch_size, cfg.block_size, rng)
        key, subkey = jax.random.split(key)
        metrics = eval_step(x, y, state, configs, subkey, eval_step_index)
        losses.append(float(metrics["loss"]))
        ce_losses.append(float(metrics["ce_loss"]))
        smoothed_ce_losses.append(float(metrics["smoothed_ce_loss"]))
        dft_losses.append(float(metrics["dft_loss"]))
        mean_gold_probs.append(float(metrics["mean_gold_prob"]))
        valid_tokens.append(float(metrics["valid_tokens"]))
        dft_alphas.append(float(metrics["dft_alpha"]))
        label_smoothings.append(float(metrics["label_smoothing"]))
    return {
        "loss": float(np.mean(losses)),
        "ce_loss": float(np.mean(ce_losses)),
        "smoothed_ce_loss": float(np.mean(smoothed_ce_losses)),
        "dft_loss": float(np.mean(dft_losses)),
        "mean_gold_prob": float(np.mean(mean_gold_probs)),
        "valid_tokens": float(np.mean(valid_tokens)),
        "dft_alpha": float(np.mean(dft_alphas)),
        "label_smoothing": float(np.mean(label_smoothings)),
    }


def estimate_loss(data: TrainingData, state, configs, key, cfg) -> float:
    return estimate_losses(data, state, configs, key, cfg)["loss"]


def meta_compat_value(meta: dict[str, Any], key: str):
    if key == "hyperconnection_mode" and key not in meta:
        return "block" if int(meta.get("hyperconnection_streams", 1)) > 1 else "none"
    if key == "hyperconnection_dynamic" and key not in meta:
        return False
    if key == "scale_token_embeddings" and key not in meta:
        return True
    if key == "final_norm" and key not in meta:
        return False
    return meta.get(key)


def model_meta_compatible(loaded_meta: dict[str, Any] | None, meta: dict[str, Any]) -> bool:
    if not isinstance(loaded_meta, dict):
        return False
    return all(meta_compat_value(loaded_meta, key) == meta_compat_value(meta, key) for key in MODEL_COMPAT_META_KEYS)


def load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def cast_floating_tree(tree, dtype=dataType):
    return jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x,
        tree,
    )


def init_optimizer(model, cfg):
    optimizer = cfg.optimizer.lower()
    if optimizer == "adamw":
        _, opt_configs = adamWOpt_init(
            model,
            lr=cfg.lr,
            beta1=cfg.adam_beta1,
            beta2=cfg.adam_beta2,
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay,
            warmup_steps=cfg.lr_warmup_steps,
            decay_steps=cfg.steps,
            min_lr_ratio=cfg.min_lr_ratio,
        )
    elif optimizer == "adam":
        _, opt_configs = adamOpt_init(model, cfg.lr, beta1=cfg.adam_beta1, beta2=cfg.adam_beta2, eps=cfg.adam_eps)
    else:
        raise ValueError(f"unsupported optimizer={cfg.optimizer!r}; expected adamw or adam")
    return opt_configs


def init_fresh_state(meta: dict[str, Any], cfg, key: jax.Array):
    key, model_key = jax.random.split(key)
    model = cast_floating_tree(build_model(model_key, meta))
    opt_configs = init_optimizer(model, cfg)
    opt_state, _ = (adamWOpt_init if cfg.optimizer.lower() == "adamw" else adamOpt_init)(
        model,
        cfg.lr,
        cfg.adam_beta1,
        cfg.adam_beta2,
        cfg.adam_eps,
    )
    return {"model": model, "opt_state": opt_state["opt_state"]}, opt_configs, key


def init_state_from_model(model, cfg):
    model = cast_floating_tree(model)
    opt_configs = init_optimizer(model, cfg)
    opt_state, _ = (adamWOpt_init if cfg.optimizer.lower() == "adamw" else adamOpt_init)(
        model,
        cfg.lr,
        cfg.adam_beta1,
        cfg.adam_beta2,
        cfg.adam_eps,
    )
    return {"model": model, "opt_state": opt_state["opt_state"]}, opt_configs


def init_or_load_state(
    checkpoint_path: Path,
    meta: dict[str, Any],
    cfg,
    key: jax.Array,
    init_checkpoint_path: Path | None = None,
):
    opt_configs = init_optimizer(None, cfg)
    start_step = 0

    if checkpoint_path.exists():
        loaded = load_checkpoint(checkpoint_path)
        if model_meta_compatible(loaded.get("meta"), meta):
            start_step = int(loaded.get("step", 0))
            model = cast_floating_tree(loaded["model"])
            if loaded.get("optimizer") == cfg.optimizer:
                state = {
                    "model": model,
                    "opt_state": cast_floating_tree(loaded["opt_state"], jnp.float32),
                }
                print(f"loaded checkpoint: {checkpoint_path} step={start_step} optimizer={cfg.optimizer}")
            else:
                state, opt_configs = init_state_from_model(model, cfg)
                print(
                    f"loaded checkpoint model: {checkpoint_path} step={start_step}; "
                    f"reset optimizer {loaded.get('optimizer')} -> {cfg.optimizer}"
                )
        else:
            print("checkpoint meta does not match current config; initializing a new model")
            state, opt_configs, key = init_fresh_state(meta, cfg, key)
    elif init_checkpoint_path is not None:
        if not init_checkpoint_path.exists():
            raise FileNotFoundError(f"init_checkpoint does not exist: {init_checkpoint_path}")
        loaded = load_checkpoint(init_checkpoint_path)
        loaded_meta = loaded.get("meta")
        if not model_meta_compatible(loaded_meta, meta):
            raise ValueError(
                f"init_checkpoint is not compatible with current model config: {init_checkpoint_path}\n"
                f"loaded_meta={loaded_meta}\ncurrent_meta={meta}"
            )
        state, opt_configs = init_state_from_model(loaded["model"], cfg)
        print(f"loaded init checkpoint model: {init_checkpoint_path}; initialized a fresh optimizer")
    else:
        state, opt_configs, key = init_fresh_state(meta, cfg, key)

    configs = StaticConfigs(
        {
            "n_heads": meta["n_heads"],
            "drop_prob": 0.0,
            "scale_token_embeddings": meta.get("scale_token_embeddings", True),
            "hyperconnection_sinkhorn_iters": meta.get("hyperconnection_sinkhorn_iters", 8),
            "attention_implementation": meta.get("attention_implementation", "cudnn"),
            "grad_clip_norm": cfg.grad_clip_norm,
            "loss_objective": normalize_loss_objective(getattr(cfg, "loss_objective", "ce")),
            "dft_alpha": getattr(cfg, "dft_alpha", 0.2),
            "dft_start_step": getattr(cfg, "dft_start_step", 0),
            "dft_warmup_steps": getattr(cfg, "dft_warmup_steps", 2000),
            "label_smoothing": getattr(cfg, "label_smoothing", 0.0),
            "meta": meta,
            **opt_configs,
        }
    )
    return state, configs, start_step, key


def save_checkpoint(path: Path, state, meta: dict[str, Any], step: int, cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {
                "model": state["model"],
                "opt_state": state["opt_state"],
                "meta": meta,
                "step": step,
                "optimizer": cfg.optimizer,
                "optimizer_config": {
                    "lr": cfg.lr,
                    "adam_beta1": cfg.adam_beta1,
                    "adam_beta2": cfg.adam_beta2,
                    "adam_eps": cfg.adam_eps,
                    "weight_decay": cfg.weight_decay,
                    "lr_warmup_steps": cfg.lr_warmup_steps,
                    "lr_decay_steps": cfg.steps,
                    "min_lr_ratio": cfg.min_lr_ratio,
                    "grad_clip_norm": cfg.grad_clip_norm,
                },
            },
            f,
        )


def jsonable_config(cfg) -> dict[str, Any]:
    payload = asdict(cfg)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def write_run_config(path: Path, cfg, meta: dict[str, Any], tokenizer: HFWordPieceTokenizer, token_count: int, config_source: str) -> None:
    payload = {
        "config_source": config_source,
        "config": jsonable_config(cfg),
        "token_count": token_count,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "meta": meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sample_next_token(logits: np.ndarray, rng: np.random.Generator, temperature: float, top_k: int) -> int:
    logits = logits.astype(np.float64)
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k > 0 and top_k < logits.shape[-1]:
        keep = np.argpartition(logits, -top_k)[-top_k:]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked
    logits = logits - np.nanmax(logits)
    probs = np.exp(logits)
    probs = probs / np.sum(probs)
    return int(rng.choice(np.arange(logits.shape[-1]), p=probs))


def generate_text(
    prompt: str,
    tokenizer: HFWordPieceTokenizer,
    state,
    configs,
    *,
    block_size: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> str:
    ids = tokenizer.encode(prompt, add_bos=True, return_np=False)
    rng = np.random.default_rng(seed)
    model = state["model"]

    for _ in range(max_new_tokens):
        context = np.array(ids[-block_size:], dtype=np.int32)
        logits = lm_apply(
            context,
            model,
            configs["n_heads"],
            -1.0,
            sinkhorn_iters=configs["hyperconnection_sinkhorn_iters"],
            scale_token_embeddings=configs["scale_token_embeddings"],
            attention_implementation=configs["attention_implementation"],
        )
        next_id = sample_next_token(np.asarray(logits[-1]), rng, temperature, top_k)
        if next_id == tokenizer.eos_id:
            break
        ids.append(next_id)

    return tokenizer.decode(ids)
