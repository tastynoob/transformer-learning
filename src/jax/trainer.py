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
    from .layer import adamOpt_init, adamWOpt_init, cross_entropy_loss_indices, dataType
    from .lm_model import build_model, lm_apply
    from .tokenizer import HFWordPieceTokenizer
except ImportError:
    from configs import StaticConfigs
    from corpus import IGNORE_INDEX, TrainingData, random_batch
    from layer import adamOpt_init, adamWOpt_init, cross_entropy_loss_indices, dataType
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


def _train_one(input_ids, target_ids, model, n_heads: int, drop_prob: float, sinkhorn_iters: int, scale_token_embeddings: bool, key):
    def compute_loss(params, subkey):
        logits = lm_apply(input_ids, params, n_heads, drop_prob, subkey, sinkhorn_iters, scale_token_embeddings)
        return cross_entropy_loss_indices(target_ids, logits, ignore_index=IGNORE_INDEX)

    loss, grads = jax.value_and_grad(compute_loss)(model, key)
    return loss, grads


def _loss_one(input_ids, target_ids, model, n_heads: int, drop_prob: float, sinkhorn_iters: int, scale_token_embeddings: bool, key):
    logits = lm_apply(input_ids, model, n_heads, drop_prob, key, sinkhorn_iters, scale_token_embeddings)
    return cross_entropy_loss_indices(target_ids, logits, ignore_index=IGNORE_INDEX)


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
def train_step(input_batch, target_batch, state, configs, key):
    model = state["model"]
    opt_state = state["opt_state"]
    batch_size = input_batch.shape[0]
    keys = jax.random.split(key, batch_size)

    losses, grads = jax.vmap(_train_one, in_axes=(0, 0, None, None, None, None, None, 0))(
        input_batch,
        target_batch,
        model,
        configs["n_heads"],
        configs["drop_prob"],
        configs["hyperconnection_sinkhorn_iters"],
        configs["scale_token_embeddings"],
        keys,
    )

    avg_loss = jnp.mean(losses)
    avg_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)
    avg_grads = clip_grads_by_global_norm(avg_grads, configs["grad_clip_norm"])

    updates, new_opt_state = configs["opt_fn"](avg_grads, opt_state, configs, model)
    new_model = jax.tree_util.tree_map(lambda p, u: p - u, model, updates)
    return {"model": new_model, "opt_state": new_opt_state}, avg_loss


@partial(jax.jit, static_argnames=("configs",))
def eval_step(input_batch, target_batch, state, configs, key):
    model = state["model"]
    batch_size = input_batch.shape[0]
    keys = jax.random.split(key, batch_size)

    losses = jax.vmap(_loss_one, in_axes=(0, 0, None, None, None, None, None, 0))(
        input_batch,
        target_batch,
        model,
        configs["n_heads"],
        -1.0,
        configs["hyperconnection_sinkhorn_iters"],
        configs["scale_token_embeddings"],
        keys,
    )
    return jnp.mean(losses)


def estimate_loss(data: TrainingData, state, configs, key, cfg) -> float:
    losses: list[float] = []
    rng = np.random.default_rng(cfg.seed + 999)
    for _ in range(cfg.eval_batches):
        x, y = random_batch(data, cfg.batch_size, cfg.block_size, rng)
        key, subkey = jax.random.split(key)
        loss = eval_step(x, y, state, configs, subkey)
        losses.append(float(loss))
    return float(np.mean(losses))


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
            "grad_clip_norm": cfg.grad_clip_norm,
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
        )
        next_id = sample_next_token(np.asarray(logits[-1]), rng, temperature, top_k)
        if next_id == tokenizer.eos_id:
            break
        ids.append(next_id)

    return tokenizer.decode(ids)
