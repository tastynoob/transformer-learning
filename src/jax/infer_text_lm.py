"""Interactive inference for the JAX text language model.

This file is intentionally separate from `text_lm.py` so inference can run
while training is still writing checkpoints. It loads the tokenizer and the
latest checkpoint, then repeatedly reads a prompt from stdin and prints the
generated continuation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import importlib.util
import os
from pathlib import Path
import pickle
import re
import sys
import time
from typing import Any

import numpy as np


def _parse_bootstrap_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-c", "--config", default=None)
    return parser.parse_known_args(argv)[0]


def _default_config_module_name() -> str:
    if __package__:
        return f"{__package__}.init"
    return "init"


def _looks_like_file_ref(config_ref: str) -> bool:
    return config_ref.endswith(".py") or "/" in config_ref or "\\" in config_ref


def _load_config_module_from_path(config_ref: str):
    path = Path(config_ref).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"config file does not exist: {path}")
    if path.suffix != ".py":
        raise ValueError(f"config file must be a .py file: {path}")

    module_name = f"_infer_text_lm_config_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load config file: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    added_paths: list[str] = []
    for import_path in (str(path.parent), str(Path.cwd())):
        if import_path not in sys.path:
            sys.path.insert(0, import_path)
            added_paths.append(import_path)
    try:
        spec.loader.exec_module(module)
    finally:
        for import_path in added_paths:
            try:
                sys.path.remove(import_path)
            except ValueError:
                pass
    return module


def _load_config_module(config_ref: str | None):
    if config_ref is None:
        return importlib.import_module(_default_config_module_name()), _default_config_module_name()
    if _looks_like_file_ref(config_ref):
        return _load_config_module_from_path(config_ref), str(Path(config_ref))
    return importlib.import_module(config_ref), config_ref


def load_train_config(config_ref: str | None = None):
    module, source = _load_config_module(config_ref)
    if not hasattr(module, "CFG"):
        raise AttributeError(f"config source {source!r} must define CFG")
    return module.CFG, source


_BOOTSTRAP_ARGS = _parse_bootstrap_args(sys.argv[1:]) if __name__ == "__main__" else argparse.Namespace(config=None)
CFG, CONFIG_SOURCE = load_train_config(_BOOTSTRAP_ARGS.config)


@dataclass
class InferenceConfig:
    tokenizer_json: Path = CFG.tokenizer_json
    checkpoint: Path = CFG.checkpoint
    init_tokenizer_json: Path | None = getattr(CFG, "init_tokenizer_json", None)
    init_checkpoint: Path | None = getattr(CFG, "init_checkpoint", None)
    config_source: str = CONFIG_SOURCE

    # Keep inference independent from the training process. CPU is enough for
    # interactive generation and avoids contending with GPU training.
    jax_platforms: str | None = "cpu"

    max_new_tokens: int = 120
    temperature: float = 0.7
    top_k: int = 20
    seed: int = 1234
    checkpoint_retries: int = 5
    checkpoint_retry_sleep: float = 1.0
    ban_unk: bool = True
    stream: bool = True
    use_kv_cache: bool = True
    chat_prompt: bool = True
    user_label: str = "User"
    assistant_label: str = "Assistant"


INFER = InferenceConfig()


if INFER.jax_platforms is not None:
    os.environ["JAX_PLATFORMS"] = INFER.jax_platforms


def _append_xla_flag(flag: str) -> None:
    flags = os.environ.get("XLA_FLAGS", "")
    if flag not in flags.split():
        os.environ["XLA_FLAGS"] = (flags + " " + flag).strip()


_append_xla_flag("--xla_gpu_autotune_level=0")

import jax
import jax.numpy as jnp

try:
    from .layer import (
        embedding_apply,
        encoderLayer_apply,
        feedforward_apply,
        linear_apply,
        multiHeadAttention_apply,
        normalize_apply,
        positionalEncoding_encode,
        positionalEncoding_init,
        split_key,
    )
    from .tokenizer import HFWordPieceTokenizer
except ImportError:
    from layer import (
        embedding_apply,
        encoderLayer_apply,
        feedforward_apply,
        linear_apply,
        multiHeadAttention_apply,
        normalize_apply,
        positionalEncoding_encode,
        positionalEncoding_init,
        split_key,
    )
    from tokenizer import HFWordPieceTokenizer


def format_interactive_prompt(prompt: str, cfg: InferenceConfig) -> str:
    if not cfg.chat_prompt:
        return prompt
    user_prefix = re.escape(cfg.user_label)
    assistant_prefix = re.escape(cfg.assistant_label)
    has_user = re.search(rf"(?im)^\s*{user_prefix}\s*:", prompt) is not None
    has_assistant = re.search(rf"(?im)^\s*{assistant_prefix}\s*:", prompt) is not None
    if has_assistant:
        return prompt
    if has_user:
        return prompt.rstrip() + f"\n{cfg.assistant_label}: "
    return f"{cfg.user_label}: {prompt.strip()}\n{cfg.assistant_label}: "


def is_block_hyperconnection_layer(layer) -> bool:
    return isinstance(layer, tuple) and len(layer) == 2 and len(layer[1]) in (3, 9)


def is_sublayer_hyperconnection_layer(layer) -> bool:
    return (
        isinstance(layer, tuple)
        and len(layer) == 3
        and isinstance(layer[1], tuple)
        and isinstance(layer[2], tuple)
        and len(layer[1]) in (3, 9)
        and len(layer[2]) in (3, 9)
    )


def hyperconnection_project_residual(res_logits, sinkhorn_iters: int):
    res_logits32 = res_logits.astype(jnp.float32)
    h = jnp.exp(res_logits32 - jnp.max(res_logits32, axis=(-2, -1), keepdims=True))
    for _ in range(sinkhorn_iters):
        h = h / jnp.maximum(jnp.sum(h, axis=-1, keepdims=True), 1e-6)
        h = h / jnp.maximum(jnp.sum(h, axis=-2, keepdims=True), 1e-6)
    return h.astype(res_logits.dtype)


def hyperconnection_read(pre_logits):
    return jax.nn.sigmoid(pre_logits)


def hyperconnection_write(post_logits):
    return 2.0 * jax.nn.sigmoid(post_logits)


def rms_normalize_last_dim(x, eps: float = 1e-6):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)


def hyperconnection_coefficients(hc_params, streams, sinkhorn_iters: int):
    if len(hc_params) == 3:
        res_logits, pre_logits, post_logits = hc_params
        return (
            hyperconnection_project_residual(res_logits, sinkhorn_iters),
            hyperconnection_read(pre_logits),
            hyperconnection_write(post_logits),
        )

    (
        res_bias,
        pre_bias,
        post_bias,
        phi_pre,
        phi_post,
        phi_res,
        alpha_pre,
        alpha_post,
        alpha_res,
    ) = hc_params
    n_streams = streams.shape[0]
    flat = streams.transpose(1, 0, 2).reshape(streams.shape[1], -1)
    flat = rms_normalize_last_dim(flat)
    pre_logits = alpha_pre * jnp.matmul(flat, phi_pre) + pre_bias
    post_logits = alpha_post * jnp.matmul(flat, phi_post) + post_bias
    res_logits = alpha_res * jnp.matmul(flat, phi_res).reshape(streams.shape[1], n_streams, n_streams) + res_bias
    return (
        hyperconnection_project_residual(res_logits, sinkhorn_iters),
        hyperconnection_read(pre_logits),
        hyperconnection_write(post_logits),
    )


def hyperconnection_mix_streams(h_res, streams):
    if h_res.ndim == 2:
        return jnp.einsum("ij,jtd->itd", h_res, streams)
    return jnp.einsum("tij,jtd->itd", h_res, streams)


def hyperconnection_read_streams(h_pre, streams):
    if h_pre.ndim == 1:
        return jnp.einsum("i,itd->td", h_pre, streams)
    return jnp.einsum("ti,itd->td", h_pre, streams)


def hyperconnection_write_streams(h_post, delta):
    if h_post.ndim == 1:
        return h_post[:, None, None] * delta[None, :, :]
    return jnp.einsum("ti,td->itd", h_post, delta)


def unpack_lm_params(params):
    if len(params) == 2:
        layers, output = params
        return layers, None, output
    if len(params) == 3:
        layers, final_norm, output = params
        return layers, final_norm, output
    raise ValueError(f"unexpected LM parameter structure with {len(params)} top-level entries")


def lm_token_embedding_apply(input_ids, output, scale_token_embeddings: bool):
    x = embedding_apply(input_ids, output[0].T)
    if not scale_token_embeddings:
        scale = jnp.sqrt(jnp.asarray(x.shape[-1], dtype=x.dtype))
        x = x / scale
    return x


def lm_head_apply(x, output, final_norm):
    if final_norm is not None:
        x = normalize_apply(x, final_norm)
    return linear_apply(x, output)


def encoderLayer_delta_apply(x, mask, params, n_heads):
    return encoderLayer_apply(x, mask, params, n_heads, -1.0, None) - x


def attention_sublayer_delta_apply(x, mask, params, n_heads):
    if len(params) == 4:
        attn, norm1, _ffn, _norm2 = params
        attn_res_scale = 1.0
    else:
        attn, norm1, _ffn, _norm2, attn_res_scale, _ffn_res_scale = params

    x_norm = normalize_apply(x, norm1)
    return attn_res_scale * multiHeadAttention_apply(x_norm, x_norm, x_norm, mask, attn, n_heads)


def feedforward_sublayer_delta_apply(x, params):
    if len(params) == 4:
        _attn, _norm1, ffn, norm2 = params
        ffn_res_scale = 1.0
    else:
        _attn, _norm1, ffn, norm2, _attn_res_scale, ffn_res_scale = params

    x_norm = normalize_apply(x, norm2)
    return ffn_res_scale * feedforward_apply(x_norm, ffn, -1.0, None)


def hyperconnection_layer_apply(streams, mask, layer, hc_params, n_heads, sinkhorn_iters: int):
    h_res, h_pre, h_post = hyperconnection_coefficients(hc_params, streams, sinkhorn_iters)
    mixed_streams = hyperconnection_mix_streams(h_res, streams)
    branch_input = hyperconnection_read_streams(h_pre, streams)
    delta = encoderLayer_delta_apply(branch_input, mask, layer, n_heads)
    return mixed_streams + hyperconnection_write_streams(h_post, delta)


def hyperconnection_attention_sublayer_apply(streams, mask, layer, hc_params, n_heads, sinkhorn_iters: int):
    h_res, h_pre, h_post = hyperconnection_coefficients(hc_params, streams, sinkhorn_iters)
    mixed_streams = hyperconnection_mix_streams(h_res, streams)
    branch_input = hyperconnection_read_streams(h_pre, streams)
    delta = attention_sublayer_delta_apply(branch_input, mask, layer, n_heads)
    return mixed_streams + hyperconnection_write_streams(h_post, delta)


def hyperconnection_feedforward_sublayer_apply(streams, layer, hc_params, sinkhorn_iters: int):
    h_res, h_pre, h_post = hyperconnection_coefficients(hc_params, streams, sinkhorn_iters)
    mixed_streams = hyperconnection_mix_streams(h_res, streams)
    branch_input = hyperconnection_read_streams(h_pre, streams)
    delta = feedforward_sublayer_delta_apply(branch_input, layer)
    return mixed_streams + hyperconnection_write_streams(h_post, delta)


def lm_apply(
    input_ids,
    params,
    n_heads: int,
    pe,
    causal_mask,
    sinkhorn_iters: int = 8,
    scale_token_embeddings: bool = True,
):
    layers, final_norm, output = unpack_lm_params(params)
    layer_keys = split_key(None, len(layers))
    x = lm_token_embedding_apply(input_ids, output, scale_token_embeddings)
    x = positionalEncoding_encode(x, pe)
    mask = causal_mask[:, : input_ids.shape[0], : input_ids.shape[0]]

    if layers and is_sublayer_hyperconnection_layer(layers[0]):
        n_streams = int(layers[0][1][0].shape[0])
        streams = jnp.broadcast_to(x, (n_streams,) + x.shape)
        for layer_with_hc in layers:
            layer, attn_hc_params, ffn_hc_params = layer_with_hc
            streams = hyperconnection_attention_sublayer_apply(
                streams,
                mask,
                layer,
                attn_hc_params,
                n_heads,
                sinkhorn_iters,
            )
            streams = hyperconnection_feedforward_sublayer_apply(streams, layer, ffn_hc_params, sinkhorn_iters)
        return lm_head_apply(jnp.mean(streams, axis=0), output, final_norm)

    if layers and is_block_hyperconnection_layer(layers[0]):
        n_streams = int(layers[0][1][0].shape[0])
        streams = jnp.broadcast_to(x, (n_streams,) + x.shape)
        for layer_with_hc in layers:
            layer, hc_params = layer_with_hc
            streams = hyperconnection_layer_apply(streams, mask, layer, hc_params, n_heads, sinkhorn_iters)
        return lm_head_apply(jnp.mean(streams, axis=0), output, final_norm)

    for layer, layer_key in zip(layers, layer_keys):
        x = encoderLayer_apply(x, mask, layer, n_heads, -1.0, layer_key)

    return lm_head_apply(x, output, final_norm)


def _empty_kv_cache(num_layers: int) -> list[tuple[jax.Array | None, jax.Array | None]]:
    return [(None, None) for _ in range(num_layers)]


def _append_kv(cache_entry, k_new, v_new):
    k_cache, v_cache = cache_entry
    if k_cache is None or v_cache is None:
        return k_new, v_new
    return jnp.concatenate([k_cache, k_new], axis=1), jnp.concatenate([v_cache, v_new], axis=1)


def cached_attention_step(x, params, cache_entry, n_heads: int):
    lin_q, lin_k, lin_v, lin_o = params
    d_k = lin_k[0].shape[1] // n_heads
    d_v = lin_v[0].shape[1] // n_heads

    q = linear_apply(x, lin_q).reshape(1, n_heads, d_k).transpose(1, 0, 2)
    k_new = linear_apply(x, lin_k).reshape(1, n_heads, d_k).transpose(1, 0, 2)
    v_new = linear_apply(x, lin_v).reshape(1, n_heads, d_v).transpose(1, 0, 2)
    k_all, v_all = _append_kv(cache_entry, k_new, v_new)

    scores = jnp.matmul(q, jnp.swapaxes(k_all, -2, -1)) / jnp.sqrt(d_k)
    attn = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(attn, v_all).transpose(1, 0, 2).reshape(1, -1)
    return linear_apply(output, lin_o), (k_all, v_all)


def cached_layer_step(x, params, cache_entry, n_heads: int):
    if len(params) == 4:
        attn, norm1, ffn, norm2 = params
        attn_res_scale = 1.0
        ffn_res_scale = 1.0
    else:
        attn, norm1, ffn, norm2, attn_res_scale, ffn_res_scale = params

    x_norm = normalize_apply(x, norm1)
    attn_out, new_cache_entry = cached_attention_step(x_norm, attn, cache_entry, n_heads)
    a = x + attn_res_scale * attn_out

    a_norm = normalize_apply(a, norm2)
    b = a + ffn_res_scale * feedforward_apply(a_norm, ffn, -1.0, None)
    return b, new_cache_entry


def cached_layer_delta_step(x, params, cache_entry, n_heads: int):
    y, new_cache_entry = cached_layer_step(x, params, cache_entry, n_heads)
    return y - x, new_cache_entry


def cached_attention_sublayer_delta_step(x, params, cache_entry, n_heads: int):
    if len(params) == 4:
        attn, norm1, _ffn, _norm2 = params
        attn_res_scale = 1.0
    else:
        attn, norm1, _ffn, _norm2, attn_res_scale, _ffn_res_scale = params

    x_norm = normalize_apply(x, norm1)
    attn_out, new_cache_entry = cached_attention_step(x_norm, attn, cache_entry, n_heads)
    return attn_res_scale * attn_out, new_cache_entry


def cached_feedforward_sublayer_delta_step(x, params):
    if len(params) == 4:
        _attn, _norm1, ffn, norm2 = params
        ffn_res_scale = 1.0
    else:
        _attn, _norm1, ffn, norm2, _attn_res_scale, ffn_res_scale = params

    x_norm = normalize_apply(x, norm2)
    return ffn_res_scale * feedforward_apply(x_norm, ffn, -1.0, None)


def cached_hyperconnection_layer_step(streams, layer, hc_params, cache_entry, n_heads: int, sinkhorn_iters: int):
    h_res, h_pre, h_post = hyperconnection_coefficients(hc_params, streams, sinkhorn_iters)
    mixed_streams = hyperconnection_mix_streams(h_res, streams)
    branch_input = hyperconnection_read_streams(h_pre, streams)
    delta, new_cache_entry = cached_layer_delta_step(branch_input, layer, cache_entry, n_heads)
    return mixed_streams + hyperconnection_write_streams(h_post, delta), new_cache_entry


def cached_hyperconnection_attention_sublayer_step(
    streams,
    layer,
    hc_params,
    cache_entry,
    n_heads: int,
    sinkhorn_iters: int,
):
    h_res, h_pre, h_post = hyperconnection_coefficients(hc_params, streams, sinkhorn_iters)
    mixed_streams = hyperconnection_mix_streams(h_res, streams)
    branch_input = hyperconnection_read_streams(h_pre, streams)
    delta, new_cache_entry = cached_attention_sublayer_delta_step(branch_input, layer, cache_entry, n_heads)
    return mixed_streams + hyperconnection_write_streams(h_post, delta), new_cache_entry


def cached_hyperconnection_feedforward_sublayer_step(streams, layer, hc_params, sinkhorn_iters: int):
    h_res, h_pre, h_post = hyperconnection_coefficients(hc_params, streams, sinkhorn_iters)
    mixed_streams = hyperconnection_mix_streams(h_res, streams)
    branch_input = hyperconnection_read_streams(h_pre, streams)
    delta = cached_feedforward_sublayer_delta_step(branch_input, layer)
    return mixed_streams + hyperconnection_write_streams(h_post, delta)


def cached_lm_step(token_id: int, pos: int, cache, params, meta: dict[str, Any], pe):
    layers, final_norm, output = unpack_lm_params(params)
    x = lm_token_embedding_apply(
        jnp.asarray([token_id], dtype=jnp.int32),
        output,
        bool(meta.get("scale_token_embeddings", True)),
    )
    x = x + pe[pos : pos + 1]

    if layers and is_sublayer_hyperconnection_layer(layers[0]):
        n_streams = int(layers[0][1][0].shape[0])
        streams = jnp.broadcast_to(x, (n_streams,) + x.shape)
        new_cache = []
        for layer_with_hc, cache_entry in zip(layers, cache):
            layer, attn_hc_params, ffn_hc_params = layer_with_hc
            streams, new_cache_entry = cached_hyperconnection_attention_sublayer_step(
                streams,
                layer,
                attn_hc_params,
                cache_entry,
                int(meta["n_heads"]),
                int(meta.get("hyperconnection_sinkhorn_iters", 8)),
            )
            streams = cached_hyperconnection_feedforward_sublayer_step(
                streams,
                layer,
                ffn_hc_params,
                int(meta.get("hyperconnection_sinkhorn_iters", 8)),
            )
            new_cache.append(new_cache_entry)
        return lm_head_apply(jnp.mean(streams, axis=0), output, final_norm), new_cache

    if layers and is_block_hyperconnection_layer(layers[0]):
        n_streams = int(layers[0][1][0].shape[0])
        streams = jnp.broadcast_to(x, (n_streams,) + x.shape)
        new_cache = []
        for layer_with_hc, cache_entry in zip(layers, cache):
            layer, hc_params = layer_with_hc
            streams, new_cache_entry = cached_hyperconnection_layer_step(
                streams,
                layer,
                hc_params,
                cache_entry,
                int(meta["n_heads"]),
                int(meta.get("hyperconnection_sinkhorn_iters", 8)),
            )
            new_cache.append(new_cache_entry)
        return lm_head_apply(jnp.mean(streams, axis=0), output, final_norm), new_cache

    new_cache = []
    for layer, cache_entry in zip(layers, cache):
        x, new_cache_entry = cached_layer_step(x, layer, cache_entry, int(meta["n_heads"]))
        new_cache.append(new_cache_entry)

    return lm_head_apply(x, output, final_norm), new_cache


def prefill_kv_cache(input_ids: list[int], model, meta: dict[str, Any], pe):
    cache = _empty_kv_cache(int(meta["n_layers"]))
    logits = None
    for pos, token_id in enumerate(input_ids):
        logits, cache = cached_lm_step(token_id, pos, cache, model, meta, pe)
    if logits is None:
        raise ValueError("cannot prefill KV cache with an empty prompt")
    return logits, cache, len(input_ids)


def load_checkpoint(path: Path, retries: int, retry_sleep: float) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with path.open("rb") as f:
                checkpoint = pickle.load(f)
            if "model" not in checkpoint or "meta" not in checkpoint:
                raise ValueError(f"{path} is not a text LM checkpoint")
            return checkpoint
        except (EOFError, pickle.UnpicklingError, OSError, ValueError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(retry_sleep)
    raise RuntimeError(f"failed to load checkpoint after {retries} attempts: {last_error}") from last_error


def resolve_runtime_paths(cfg: InferenceConfig) -> tuple[Path, Path, str, str]:
    checkpoint = cfg.checkpoint
    checkpoint_source = "checkpoint"
    if not checkpoint.exists() and cfg.init_checkpoint is not None:
        checkpoint = cfg.init_checkpoint
        checkpoint_source = "init_checkpoint"

    tokenizer_json = cfg.tokenizer_json
    tokenizer_source = "tokenizer_json"
    if checkpoint_source == "init_checkpoint" and cfg.init_tokenizer_json is not None:
        tokenizer_json = cfg.init_tokenizer_json
        tokenizer_source = "init_tokenizer_json"
    elif not tokenizer_json.exists() and cfg.init_tokenizer_json is not None:
        tokenizer_json = cfg.init_tokenizer_json
        tokenizer_source = "init_tokenizer_json"

    if not tokenizer_json.exists():
        raise FileNotFoundError(
            f"tokenizer not found: {tokenizer_json}. "
            f"config={cfg.config_source} tokenizer_json={cfg.tokenizer_json} "
            f"init_tokenizer_json={cfg.init_tokenizer_json}"
        )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}. "
            f"config={cfg.config_source} checkpoint={cfg.checkpoint} init_checkpoint={cfg.init_checkpoint}"
        )

    return tokenizer_json, checkpoint, tokenizer_source, checkpoint_source


def load_runtime(cfg: InferenceConfig):
    tokenizer_json, checkpoint_path, tokenizer_source, checkpoint_source = resolve_runtime_paths(cfg)

    tokenizer = HFWordPieceTokenizer.load(tokenizer_json)
    checkpoint = load_checkpoint(checkpoint_path, cfg.checkpoint_retries, cfg.checkpoint_retry_sleep)
    meta = checkpoint["meta"]
    model = checkpoint["model"]

    if tokenizer.vocab_size != meta["vocab_size"]:
        raise ValueError(
            f"tokenizer vocab_size={tokenizer.vocab_size} does not match checkpoint vocab_size={meta['vocab_size']}"
        )

    pe = positionalEncoding_init(int(meta["max_seqlen"]), int(meta["d_model"]))
    causal_mask = jnp.tril(jnp.ones((int(meta["max_seqlen"]), int(meta["max_seqlen"]))))
    causal_mask = causal_mask[jnp.newaxis, :, :]

    runtime_info = {
        "tokenizer_json": tokenizer_json,
        "checkpoint": checkpoint_path,
        "tokenizer_source": tokenizer_source,
        "checkpoint_source": checkpoint_source,
    }
    return tokenizer, model, meta, pe, causal_mask, int(checkpoint.get("step", 0)), runtime_info


def sample_next_token(logits: np.ndarray, rng: np.random.Generator, tokenizer: HFWordPieceTokenizer, cfg: InferenceConfig) -> int:
    logits = logits.astype(np.float64)
    logits[tokenizer.pad_id] = -np.inf
    logits[tokenizer.bos_id] = -np.inf
    if cfg.ban_unk:
        logits[tokenizer.unk_id] = -np.inf

    if cfg.temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / cfg.temperature
    if cfg.top_k > 0 and cfg.top_k < logits.shape[-1]:
        keep = np.argpartition(logits, -cfg.top_k)[-cfg.top_k:]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked

    logits = logits - np.nanmax(logits)
    probs = np.exp(logits)
    probs = probs / np.sum(probs)
    return int(rng.choice(np.arange(logits.shape[-1]), p=probs))


def generate_stream(prompt: str, tokenizer: HFWordPieceTokenizer, model, meta: dict[str, Any], pe, causal_mask, cfg: InferenceConfig):
    prompt = format_interactive_prompt(prompt, cfg)
    if cfg.use_kv_cache:
        yield from generate_stream_cached(prompt, tokenizer, model, meta, pe, cfg)
        return

    ids = tokenizer.encode(prompt, add_bos=True, return_np=False)
    rng = np.random.default_rng(cfg.seed)
    block_size = int(meta["max_seqlen"])
    n_heads = int(meta["n_heads"])

    for _ in range(cfg.max_new_tokens):
        context = np.array(ids[-block_size:], dtype=np.int32)
        logits = lm_apply(
            context,
            model,
            n_heads,
            pe,
            causal_mask,
            int(meta.get("hyperconnection_sinkhorn_iters", 8)),
            bool(meta.get("scale_token_embeddings", True)),
        )
        next_id = sample_next_token(np.asarray(logits[-1]), rng, tokenizer, cfg)
        if next_id == tokenizer.eos_id:
            break
        ids.append(next_id)
        yield tokenizer.decode([next_id])


def generate_stream_cached(prompt: str, tokenizer: HFWordPieceTokenizer, model, meta: dict[str, Any], pe, cfg: InferenceConfig):
    ids = tokenizer.encode(prompt, add_bos=True, return_np=False)
    rng = np.random.default_rng(cfg.seed)
    block_size = int(meta["max_seqlen"])

    context = ids[-block_size:]
    logits, cache, cache_len = prefill_kv_cache(context, model, meta, pe)

    for _ in range(cfg.max_new_tokens):
        next_id = sample_next_token(np.asarray(logits[-1]), rng, tokenizer, cfg)
        if next_id == tokenizer.eos_id:
            break
        ids.append(next_id)
        yield tokenizer.decode([next_id])

        if cache_len >= block_size:
            context = ids[-block_size:]
            logits, cache, cache_len = prefill_kv_cache(context, model, meta, pe)
        else:
            logits, cache = cached_lm_step(next_id, cache_len, cache, model, meta, pe)
            cache_len += 1


def generate(prompt: str, tokenizer: HFWordPieceTokenizer, model, meta: dict[str, Any], pe, causal_mask, cfg: InferenceConfig) -> str:
    return "".join(generate_stream(prompt, tokenizer, model, meta, pe, causal_mask, cfg))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive inference for the JAX text language model.")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Python module name or .py file that defines CFG. Example: --config src/jax/init.py",
    )
    return parser.parse_args()


def interactive_loop() -> None:
    tokenizer, model, meta, pe, causal_mask, step, runtime_info = load_runtime(INFER)
    print(f"config: {INFER.config_source}")
    print(f"loaded checkpoint: {runtime_info['checkpoint']} step={step} source={runtime_info['checkpoint_source']}")
    print(
        f"loaded tokenizer: {runtime_info['tokenizer_json']} "
        f"vocab_size={tokenizer.vocab_size} source={runtime_info['tokenizer_source']}"
    )
    if INFER.chat_prompt:
        print(f"prompt mode: auto chat wrapper ({INFER.user_label}: ... / {INFER.assistant_label}:)")
    else:
        print("prompt mode: raw")
    print("commands: /reload reload checkpoint, /config show config, /exit quit")

    while True:
        try:
            prompt = input("\n输入> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            break
        if prompt == "/config":
            print(INFER)
            print(f"runtime={runtime_info}")
            print(f"meta={meta}")
            continue
        if prompt == "/reload":
            tokenizer, model, meta, pe, causal_mask, step, runtime_info = load_runtime(INFER)
            print(f"reloaded checkpoint: {runtime_info['checkpoint']} step={step}")
            continue

        print("输出> ", end="", flush=True)
        if INFER.stream:
            for piece in generate_stream(prompt, tokenizer, model, meta, pe, causal_mask, INFER):
                print(piece, end="", flush=True)
            print()
        else:
            print(generate(prompt, tokenizer, model, meta, pe, causal_mask, INFER), flush=True)


if __name__ == "__main__":
    parse_args()
    interactive_loop()
