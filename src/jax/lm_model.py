"""Decoder-only language model architecture."""

from __future__ import annotations

from typing import Any
import math

import jax
import jax.numpy as jnp

try:
    from .layer import (
        dataType,
        embedding_apply,
        encoderLayer_apply,
        encoderLayer_init,
        feedforward_apply,
        linear_apply,
        multiHeadAttention_apply,
        normalize_apply,
        normalize_init,
        positionalEncoding_encode,
        positionalEncoding_init,
        split_key,
        tied_embedding_output_init,
    )
except ImportError:
    from layer import (
        dataType,
        embedding_apply,
        encoderLayer_apply,
        encoderLayer_init,
        feedforward_apply,
        linear_apply,
        multiHeadAttention_apply,
        normalize_apply,
        normalize_init,
        positionalEncoding_encode,
        positionalEncoding_init,
        split_key,
        tied_embedding_output_init,
    )


def build_meta(
    *,
    vocab_size: int,
    max_seqlen: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    n_layers: int,
    scale_token_embeddings: bool,
    final_norm: bool,
    hyperconnection_streams: int,
    hyperconnection_mode: str | None = None,
    hyperconnection_dynamic: bool | None = None,
    hyperconnection_sinkhorn_iters: int,
) -> dict[str, Any]:
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if hyperconnection_streams < 1:
        raise ValueError("hyperconnection_streams must be at least 1")
    if hyperconnection_streams == 1:
        resolved_hyperconnection_mode = "none"
        resolved_hyperconnection_dynamic = False
    else:
        resolved_hyperconnection_mode = hyperconnection_mode or "sublayer"
        if resolved_hyperconnection_mode not in {"block", "sublayer"}:
            raise ValueError("hyperconnection_mode must be one of: None, 'block', 'sublayer'")
        resolved_hyperconnection_dynamic = (
            resolved_hyperconnection_mode == "sublayer"
            if hyperconnection_dynamic is None
            else bool(hyperconnection_dynamic)
        )
        if resolved_hyperconnection_mode != "sublayer":
            resolved_hyperconnection_dynamic = False
    return {
        "version": 1,
        "kind": "causal_text_lm",
        "vocab_size": vocab_size,
        "max_seqlen": max_seqlen,
        "d_model": d_model,
        "d_ff": d_ff,
        "d_k": d_model // n_heads,
        "d_v": d_model // n_heads,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "scale_token_embeddings": scale_token_embeddings,
        "final_norm": final_norm,
        "hyperconnection_streams": hyperconnection_streams,
        "hyperconnection_mode": resolved_hyperconnection_mode,
        "hyperconnection_dynamic": resolved_hyperconnection_dynamic,
        "hyperconnection_sinkhorn_iters": hyperconnection_sinkhorn_iters,
    }


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def hyperconnection_init(key: jax.Array, n_streams: int, d_model: int, dynamic: bool = True):
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    eye = jnp.eye(n_streams, dtype=jnp.float32)
    res_logits = eye * 4.0 + jax.random.normal(k1, (n_streams, n_streams), dtype=jnp.float32) * 0.01
    pre_base = _logit(1.0 / n_streams)
    pre_logits = jnp.full((n_streams,), pre_base, dtype=jnp.float32)
    pre_logits = pre_logits + jax.random.normal(k2, (n_streams,), dtype=jnp.float32) * 0.01
    post_logits = jax.random.normal(k3, (n_streams,), dtype=jnp.float32) * 0.01
    if not dynamic:
        return res_logits.astype(dataType), pre_logits.astype(dataType), post_logits.astype(dataType)

    flat_dim = n_streams * d_model
    phi_std = jnp.sqrt(1.0 / flat_dim)
    phi_pre = jax.random.normal(k4, (flat_dim, n_streams), dtype=jnp.float32) * phi_std
    phi_post = jax.random.normal(k5, (flat_dim, n_streams), dtype=jnp.float32) * phi_std
    phi_res = jax.random.normal(k6, (flat_dim, n_streams * n_streams), dtype=jnp.float32) * phi_std
    alpha_pre = jnp.asarray(1e-3, dtype=dataType)
    alpha_post = jnp.asarray(1e-3, dtype=dataType)
    alpha_res = jnp.asarray(1e-3, dtype=dataType)
    return (
        res_logits.astype(dataType),
        pre_logits.astype(dataType),
        post_logits.astype(dataType),
        phi_pre.astype(dataType),
        phi_post.astype(dataType),
        phi_res.astype(dataType),
        alpha_pre,
        alpha_post,
        alpha_res,
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


def encoderLayer_delta_apply(x, mask, params, n_heads, drop_prob=0.1, key=None):
    return encoderLayer_apply(x, mask, params, n_heads, drop_prob, key) - x


def attention_sublayer_delta_apply(x, mask, params, n_heads):
    if len(params) == 4:
        attn, norm1, _ffn, _norm2 = params
        attn_res_scale = 1.0
    else:
        attn, norm1, _ffn, _norm2, attn_res_scale, _ffn_res_scale = params

    x_norm = normalize_apply(x, norm1)
    return attn_res_scale * multiHeadAttention_apply(x_norm, x_norm, x_norm, mask, attn, n_heads)


def feedforward_sublayer_delta_apply(x, params, drop_prob=0.1, key=None):
    if len(params) == 4:
        _attn, _norm1, ffn, norm2 = params
        ffn_res_scale = 1.0
    else:
        _attn, _norm1, ffn, norm2, _attn_res_scale, ffn_res_scale = params

    x_norm = normalize_apply(x, norm2)
    return ffn_res_scale * feedforward_apply(x_norm, ffn, drop_prob, key)


def hyperconnection_layer_apply(streams, mask, layer, hc_params, n_heads, drop_prob, key, sinkhorn_iters: int):
    h_res, h_pre, h_post = hyperconnection_coefficients(hc_params, streams, sinkhorn_iters)
    mixed_streams = hyperconnection_mix_streams(h_res, streams)
    branch_input = hyperconnection_read_streams(h_pre, streams)
    delta = encoderLayer_delta_apply(branch_input, mask, layer, n_heads, drop_prob, key)
    return mixed_streams + hyperconnection_write_streams(h_post, delta)


def hyperconnection_attention_sublayer_apply(streams, mask, layer, hc_params, n_heads, sinkhorn_iters: int):
    h_res, h_pre, h_post = hyperconnection_coefficients(hc_params, streams, sinkhorn_iters)
    mixed_streams = hyperconnection_mix_streams(h_res, streams)
    branch_input = hyperconnection_read_streams(h_pre, streams)
    delta = attention_sublayer_delta_apply(branch_input, mask, layer, n_heads)
    return mixed_streams + hyperconnection_write_streams(h_post, delta)


def hyperconnection_feedforward_sublayer_apply(streams, layer, hc_params, drop_prob, key, sinkhorn_iters: int):
    h_res, h_pre, h_post = hyperconnection_coefficients(hc_params, streams, sinkhorn_iters)
    mixed_streams = hyperconnection_mix_streams(h_res, streams)
    branch_input = hyperconnection_read_streams(h_pre, streams)
    delta = feedforward_sublayer_delta_apply(branch_input, layer, drop_prob, key)
    return mixed_streams + hyperconnection_write_streams(h_post, delta)


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


def build_model(key: jax.Array, meta: dict[str, Any]):
    n_streams = int(meta.get("hyperconnection_streams", 1))
    hyperconnection_mode = meta.get("hyperconnection_mode", "block" if n_streams > 1 else "none")
    hyperconnection_dynamic = bool(meta.get("hyperconnection_dynamic", False))
    keys_per_layer = 3 if n_streams > 1 and hyperconnection_mode == "sublayer" else 2
    keys = jax.random.split(key, meta["n_layers"] * keys_per_layer + 1)
    layers = []
    for i in range(meta["n_layers"]):
        key_base = i * keys_per_layer
        layer = encoderLayer_init(
            keys[key_base],
            meta["d_model"],
            meta["d_ff"],
            meta["d_k"],
            meta["d_v"],
            meta["n_heads"],
        )
        if n_streams > 1 and hyperconnection_mode == "sublayer":
            layer = (
                layer,
                hyperconnection_init(keys[key_base + 1], n_streams, meta["d_model"], hyperconnection_dynamic),
                hyperconnection_init(keys[key_base + 2], n_streams, meta["d_model"], hyperconnection_dynamic),
            )
        elif n_streams > 1:
            layer = (layer, hyperconnection_init(keys[key_base + 1], n_streams, meta["d_model"], False))
        layers.append(layer)
    output = tied_embedding_output_init(keys[-1], meta["d_model"], meta["vocab_size"])
    if bool(meta.get("final_norm", False)):
        return tuple(layers), normalize_init(meta["d_model"]), output
    return tuple(layers), output


def lm_apply(
    input_ids,
    params,
    n_heads: int,
    drop_prob: float,
    key=None,
    sinkhorn_iters: int = 8,
    scale_token_embeddings: bool = True,
):
    layers, final_norm, output = unpack_lm_params(params)
    layer_keys = split_key(key, len(layers))
    x = lm_token_embedding_apply(input_ids, output, scale_token_embeddings)
    pe = positionalEncoding_init(input_ids.shape[0], x.shape[-1])
    x = positionalEncoding_encode(x, pe)
    causal_mask = jnp.tril(jnp.ones((input_ids.shape[0], input_ids.shape[0])))
    causal_mask = causal_mask[jnp.newaxis, :, :]

    if layers and is_sublayer_hyperconnection_layer(layers[0]):
        n_streams = int(layers[0][1][0].shape[0])
        streams = jnp.broadcast_to(x, (n_streams,) + x.shape)
        for layer_with_hc, layer_key in zip(layers, layer_keys):
            layer, attn_hc_params, ffn_hc_params = layer_with_hc
            _attn_key, ffn_key = split_key(layer_key, 2)
            streams = hyperconnection_attention_sublayer_apply(
                streams,
                causal_mask,
                layer,
                attn_hc_params,
                n_heads,
                sinkhorn_iters,
            )
            streams = hyperconnection_feedforward_sublayer_apply(
                streams,
                layer,
                ffn_hc_params,
                drop_prob,
                ffn_key,
                sinkhorn_iters,
            )
        return lm_head_apply(jnp.mean(streams, axis=0), output, final_norm)

    if layers and is_block_hyperconnection_layer(layers[0]):
        n_streams = int(layers[0][1][0].shape[0])
        streams = jnp.broadcast_to(x, (n_streams,) + x.shape)
        for layer_with_hc, layer_key in zip(layers, layer_keys):
            layer, hc_params = layer_with_hc
            streams = hyperconnection_layer_apply(
                streams,
                causal_mask,
                layer,
                hc_params,
                n_heads,
                drop_prob,
                layer_key,
                sinkhorn_iters,
            )
        return lm_head_apply(jnp.mean(streams, axis=0), output, final_norm)

    for layer, layer_key in zip(layers, layer_keys):
        x = encoderLayer_apply(x, causal_mask, layer, n_heads, drop_prob, layer_key)

    return lm_head_apply(x, output, final_norm)
