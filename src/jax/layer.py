import jax
import jax.numpy as jnp

dataType = jnp.float16
tied_embedding_std = 0.02


def split_key(key, num):
    """按需切分 PRNGKey；当 key 为 None 时返回一组 None。"""
    if key is None:
        return (None,) * num
    return tuple(jax.random.split(key, num))

def linear_init(key, in_cols, out_cols, with_bias=False):
    """初始化线性层的参数 (权重和可选的偏置)"""
    k1, k2 = jax.random.split(key)
    weight = jax.random.normal(k1, (in_cols, out_cols), dtype=dataType) * jnp.sqrt(2.0 / in_cols)
    if with_bias:
        bias = jnp.zeros((out_cols,))
        return weight, bias
    else:
        return (weight,)


def tied_embedding_output_init(key, d_model, vocab_size, std=tied_embedding_std):
    """初始化共享的 embedding / output projection 权重。"""
    weight = jax.random.normal(key, (d_model, vocab_size), dtype=dataType) * std
    return (weight,)

def linear_apply(x, params):
    """线性层的前向传播"""
    weight = params[0]
    t = jnp.dot(x, weight)
    # 如果有偏置，添加偏置
    if len(params) > 1:
        bias = params[1]
        return t + bias
    else:
        return t

def feedforward_init(key, in_cols, hidden_cols, out_cols, with_bias=False):
    """初始化前馈神经网络的参数"""
    k1, k2 = jax.random.split(key)
    lin1 = linear_init(k1, in_cols, hidden_cols, with_bias)
    lin2 = linear_init(k2, hidden_cols, out_cols, with_bias)
    return lin1, lin2

def feedforward_apply(x, params, drop_prob=0.0, key=None):
    """前馈神经网络的前向传播"""
    lin1, lin2 = params
    t = linear_apply(x, lin1)
    t = relu_apply(t)
    t = dropout_apply(t, drop_prob, key)
    t = linear_apply(t, lin2)
    return t


def dropout_apply(x, drop_prob, key):
    """Dropout 层的前向传播"""
    if drop_prob > 0.0 and key is not None:
        keep_prob = 1.0 - drop_prob
        mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)
        return jnp.where(mask, x / keep_prob, 0)
    else:
        return x

def normalize_init(dim):
    """初始化 LayerNorm 层的参数"""
    gamma = jnp.ones((dim,), dtype=dataType)
    beta = jnp.zeros((dim,), dtype=dataType)
    return gamma, beta

def normalize_apply(x, params, eps=1e-5):
    """LayerNorm 层的前向传播"""
    gamma, beta = params
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    normalized_x = (x - mean) / jnp.sqrt(variance + eps)
    return gamma * normalized_x + beta

def softmax_apply(x):
    """Softmax 激活函数"""
    return jax.nn.softmax(x, axis=-1)

def relu_apply(x):
    """ReLU 激活函数"""
    return jnp.maximum(0, x)

#  attention

def scaledDotProduct(q, k, v, mask=None):
    """缩放点积注意力机制"""
    d_k = q.shape[-1]
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d_k)
    # scores = jnp.clip(scores, -5, 5)  # 防止数值不稳定
    if mask is not None:
        scores = jnp.where(mask == 0, -1e9, scores)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(attn_weights, v)
    return output

def selfAttention_init(key, d_model, d_k, d_v):
    """初始化自注意力机制的参数"""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    lin_q = linear_init(k1, d_model, d_k)
    lin_k = linear_init(k2, d_model, d_k)
    lin_v = linear_init(k3, d_model, d_v)
    lin_o = linear_init(k4, d_v, d_model)
    return lin_q, lin_k, lin_v, lin_o

def selfAttention_apply(x, mask, params):
    """自注意力机制的前向传播"""
    lin_q, lin_k, lin_v, lin_o = params
    q = linear_apply(x, lin_q)
    k = linear_apply(x, lin_k)
    v = linear_apply(x, lin_v)
    attn_output = scaledDotProduct(q, k, v, mask)
    output = linear_apply(attn_output, lin_o)
    return output

def multiHeadAttention_init(key, d_model, d_k, d_v, n_heads):
    """初始化多头注意力机制的参数"""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    lin_q = linear_init(k1, d_model, d_k * n_heads)
    lin_k = linear_init(k2, d_model, d_k * n_heads)
    lin_v = linear_init(k3, d_model, d_v * n_heads)
    lin_o = linear_init(k4, d_v * n_heads, d_model)
    return lin_q, lin_k, lin_v, lin_o

def multiHeadAttention_apply(q, k, v, mask, params, n_heads):
    """多头注意力机制的前向传播"""
    '''
    q : (seqlen, q_dim)
    k : (seqlen, k_dim)
    v : (seqlen, v_dim)
    mask : (seqlen, seqlen)
    '''
    lin_q, lin_k, lin_v, lin_o = params
    q_seqlen = q.shape[0]
    k_seqlen = k.shape[0]
    v_seqlen = v.shape[0]
    d_k = (lin_k[0].shape[1] // n_heads)
    d_v = (lin_v[0].shape[1] // n_heads)

    q_proj = linear_apply(q, lin_q)
    k_proj = linear_apply(k, lin_k)
    v_proj = linear_apply(v, lin_v)

    q_heads = q_proj.reshape(q_seqlen, n_heads, d_k).transpose(1, 0, 2)
    k_heads = k_proj.reshape(k_seqlen, n_heads, d_k).transpose(1, 0, 2)
    v_heads = v_proj.reshape(v_seqlen, n_heads, d_v).transpose(1, 0, 2)

    attn_output = scaledDotProduct(q_heads, k_heads, v_heads, mask)

    concat_attn = attn_output.transpose(1, 0, 2).reshape(q_seqlen, -1)
    
    output = linear_apply(concat_attn, lin_o)
    return output

def encoderLayer_init(key, d_model, d_ff, d_k, d_v, n_heads):
    k_attn, k_ffn = split_key(key, 2)
    attn = multiHeadAttention_init(k_attn, d_model, d_k, d_v, n_heads)
    norm1 = normalize_init(d_model)
    ffn = feedforward_init(k_ffn, d_model, d_ff, d_model)
    norm2 = normalize_init(d_model)
    return attn, norm1, ffn, norm2

def encoderLayer_apply(x, mask, params, n_heads, drop_prob=0.1, key=None):
    if len(params) == 4:
        attn, norm1, ffn, norm2 = params
        attn_res_scale = 1.0
        ffn_res_scale = 1.0
    else:
        attn, norm1, ffn, norm2, attn_res_scale, ffn_res_scale = params

    x_norm = normalize_apply(x, norm1)
    a = multiHeadAttention_apply(x_norm, x_norm, x_norm, mask, attn, n_heads)
    a = x + attn_res_scale * a

    a_norm = normalize_apply(a, norm2)
    b = feedforward_apply(a_norm, ffn, drop_prob, key)
    b = a + ffn_res_scale * b
    return b

def decoderLayer_init(key, d_model, d_ff, d_k, d_v, n_heads):
    k1, k2, k3 = split_key(key, 3)
    self_attn = multiHeadAttention_init(k1, d_model, d_k, d_v, n_heads)
    norm1 = normalize_init(d_model)
    cross_attn = multiHeadAttention_init(k2, d_model, d_k, d_v, n_heads)
    norm2 = normalize_init(d_model)
    ffn = feedforward_init(k3, d_model, d_ff, d_model)
    norm3 = normalize_init(d_model)
    return self_attn, norm1, cross_attn, norm2, ffn, norm3

def decoderLayer_apply(x, enc_output, lookahead_mask, enc_padding_mask, params, n_heads, drop_prob=0.1, key=None):
    if len(params) == 6:
        self_attn, norm1, cross_attn, norm2, ffn, norm3 = params
        self_attn_res_scale = 1.0
        cross_attn_res_scale = 1.0
        ffn_res_scale = 1.0
    else:
        self_attn, norm1, cross_attn, norm2, ffn, norm3, self_attn_res_scale, cross_attn_res_scale, ffn_res_scale = params

    x_norm = normalize_apply(x, norm1)
    a = multiHeadAttention_apply(x_norm, x_norm, x_norm, lookahead_mask, self_attn, n_heads)
    a = x + self_attn_res_scale * a

    a_norm = normalize_apply(a, norm2)
    b = multiHeadAttention_apply(a_norm, enc_output, enc_output, enc_padding_mask, cross_attn, n_heads)
    b = a + cross_attn_res_scale * b

    b_norm = normalize_apply(b, norm3)
    c = feedforward_apply(b_norm, ffn, drop_prob, key)
    c = b + ffn_res_scale * c
    return c

def embedding_init(key, vocab_size, d_model):
    """初始化嵌入层"""
    embedding_matrix = jax.random.normal(key, (vocab_size, d_model), dtype=dataType) * 0.01
    return embedding_matrix

def embedding_apply(x, matrix):
    """嵌入层的前向传播"""
    scale = jnp.sqrt(matrix.shape[1])
    return matrix[x] * scale

def pading_mask(seq, pad_id=0):
    """生成填充掩码"""
    seq = jnp.asarray(seq)
    valid_tokens = (seq != pad_id)
    if seq.ndim == 1:
        return valid_tokens[jnp.newaxis, jnp.newaxis, :]
    if seq.ndim == 2:
        return valid_tokens[:, jnp.newaxis, jnp.newaxis, :]
    raise ValueError("pading_mask expects a 1D or 2D token array.")


def padding_mask(seq, pad_id=0):
    return pading_mask(seq, pad_id)

def positionalEncoding_init(max_len, d_model):
    """初始化位置编码"""
    position = jnp.arange(max_len)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
    pe = jnp.zeros((max_len, d_model), dtype=dataType)
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe

def positionalEncoding_encode(x, pe):
    """添加位置编码到输入"""
    seq_len = x.shape[0]
    return x + pe[:seq_len]

def adaptiveOpt_init(model, lr=0.01, eps=1e-8):
    """自适应梯度下降优化器的初始化"""
    if model is not None:
        acc = jax.tree_util.tree_map(jnp.zeros_like, model)
    else:
        acc = None

    # 可训练的参数
    params = {
        'opt_state' : acc,
    }
    # 静态参数
    configs = {
        'lr' : lr,
        'eps' : eps,
        'opt_fn' : adaptiveOpt_update
    }
    return params, configs

def adaptiveOpt_update(grads, opt_state, configs, params=None):
    """自适应梯度下降优化器的更新"""
    acc = opt_state
    lr = configs['lr']
    eps = configs['eps']

    acc = jax.tree_util.tree_map(
        lambda a, g: a + jnp.square(g),
        acc,
        grads
    )
    updates = jax.tree_util.tree_map(
        lambda g, a: lr * g / (jnp.sqrt(a) + eps),
        grads,
        acc
    )
    return updates, acc

def adamOpt_init(model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam 优化器的初始化"""
    if model is not None:
        m = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x, dtype=jnp.float32), model)
        v = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x, dtype=jnp.float32), model)
    else:
        m, v = None, None

    # 可训练的参数
    params = {
        'opt_state': {
            'm' : m,
            'v' : v,
            't' : 0
        }
    }
    # 静态参数
    configs = {
        'lr' : lr,
        'beta1' : beta1,
        'beta2' : beta2,
        'eps' : eps,
        'opt_fn' : adamOpt_update
    }
    return params, configs

def adamOpt_update(grads, opt_state, configs, params=None):
    """Adam 优化器的更新"""
    m = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), opt_state['m'])
    v = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), opt_state['v'])
    t = opt_state['t'] + 1

    beta1 = configs['beta1']
    beta2 = configs['beta2']
    lr = configs['lr']
    eps = configs['eps']

    m = jax.tree_util.tree_map(
        lambda m, g: beta1 * m + (1 - beta1) * g.astype(jnp.float32),
        m,
        grads
    )
    v = jax.tree_util.tree_map(
        lambda v, g: beta2 * v + (1 - beta2) * jnp.square(g.astype(jnp.float32)),
        v,
        grads
    )
    m_hat = jax.tree_util.tree_map(
        lambda m: m / (1 - jnp.power(beta1, t)),
        m
    )
    v_hat = jax.tree_util.tree_map(
        lambda v: v / (1 - jnp.power(beta2, t)),
        v
    )
    updates = jax.tree_util.tree_map(
        lambda m_h, v_h: lr * m_h / (jnp.sqrt(v_h) + eps),
        m_hat,
        v_hat
    )
    new_opt_state = {
        'm' : m,
        'v' : v,
        't' : t
    }
    return updates, new_opt_state


def adamWOpt_init(
    model,
    lr=0.001,
    beta1=0.9,
    beta2=0.95,
    eps=1e-8,
    weight_decay=0.01,
    warmup_steps=0,
    decay_steps=0,
    min_lr_ratio=0.1,
):
    """AdamW optimizer with fp32 moments and optional warmup/cosine decay."""
    if model is not None:
        m = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), model)
        v = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), model)
    else:
        m, v = None, None

    params = {
        'opt_state': {
            'm': m,
            'v': v,
            't': 0,
        }
    }
    configs = {
        'lr': lr,
        'beta1': beta1,
        'beta2': beta2,
        'eps': eps,
        'weight_decay': weight_decay,
        'lr_warmup_steps': warmup_steps,
        'lr_decay_steps': decay_steps,
        'min_lr_ratio': min_lr_ratio,
        'opt_fn': adamWOpt_update,
    }
    return params, configs


def _scheduled_lr(configs, step):
    lr = jnp.asarray(configs['lr'], dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)

    warmup_steps = int(configs['lr_warmup_steps'])
    if warmup_steps > 0:
        warmup = jnp.minimum(1.0, step_f / float(warmup_steps))
    else:
        warmup = jnp.asarray(1.0, dtype=jnp.float32)

    decay_steps = int(configs['lr_decay_steps'])
    min_lr_ratio = float(configs['min_lr_ratio'])
    if decay_steps > warmup_steps:
        progress = (step_f - float(warmup_steps)) / float(decay_steps - warmup_steps)
        progress = jnp.clip(progress, 0.0, 1.0)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        decay = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    else:
        decay = jnp.asarray(1.0, dtype=jnp.float32)

    return lr * jnp.minimum(warmup, decay)


def adamWOpt_update(grads, opt_state, configs, params=None):
    if params is None:
        raise ValueError("adamWOpt_update requires current params for decoupled weight decay")

    m = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), opt_state['m'])
    v = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), opt_state['v'])
    t = opt_state['t'] + 1

    beta1 = configs['beta1']
    beta2 = configs['beta2']
    eps = configs['eps']
    weight_decay = configs['weight_decay']
    lr_t = _scheduled_lr(configs, t)

    grads32 = jax.tree_util.tree_map(lambda g: g.astype(jnp.float32), grads)
    params32 = jax.tree_util.tree_map(lambda p: p.astype(jnp.float32), params)

    m = jax.tree_util.tree_map(
        lambda m, g: beta1 * m + (1 - beta1) * g,
        m,
        grads32,
    )
    v = jax.tree_util.tree_map(
        lambda v, g: beta2 * v + (1 - beta2) * jnp.square(g),
        v,
        grads32,
    )
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - jnp.power(beta1, t)), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - jnp.power(beta2, t)), v)

    updates = jax.tree_util.tree_map(
        lambda m_h, v_h, p: lr_t * (
            m_h / (jnp.sqrt(v_h) + eps) + (weight_decay * p if p.ndim > 1 else 0.0)
        ),
        m_hat,
        v_hat,
        params32,
    )
    new_opt_state = {
        'm': m,
        'v': v,
        't': t,
    }
    return updates, new_opt_state


def absolute_loss(y_true, y_pred):
    """绝对值损失函数"""
    return jnp.mean(jnp.abs(y_true - y_pred))

def mean_squared_loss(y_true, y_pred):
    """均方误差损失函数"""
    return jnp.mean((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true, y_pred):
    eps = 1e-7
    log_likelihood = jnp.sum(y_true * jnp.log(y_pred + eps), axis=-1)
    return -jnp.mean(log_likelihood)

def cross_entropy_loss_indices(y_true_indices, logits, ignore_index=None):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    if ignore_index is not None:
        safe_indices = jnp.where(y_true_indices == ignore_index, 0, y_true_indices)
    else:
        safe_indices = y_true_indices
    log_likelihood = jnp.take_along_axis(log_probs, safe_indices[..., jnp.newaxis], axis=-1)
    log_likelihood = jnp.squeeze(log_likelihood, axis=-1)

    if ignore_index is None:
        return -jnp.mean(log_likelihood)

    mask = (y_true_indices != ignore_index).astype(logits.dtype)
    num_valid = jnp.sum(mask)
    masked_loss = -jnp.sum(log_likelihood * mask) / jnp.maximum(num_valid, 1.0)
    return jnp.where(num_valid > 0, masked_loss, 0.0)
