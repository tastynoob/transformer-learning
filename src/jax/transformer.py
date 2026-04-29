from functools import partial
import os
import time
import sys
import random
import numpy as np
import tqdm
import pickle
from configs import *

use_cpu = False

if use_cpu:
    os.environ['JAX_PLATFORMS'] = 'cpu'

def append_xla_flag(flag):
    flags = os.environ.get('XLA_FLAGS', '')
    if flag not in flags.split():
        os.environ['XLA_FLAGS'] = (flags + ' ' + flag).strip()

# ROCm/gfx1150 currently fails while loading XLA's autotune redzone helper
# kernel (RepeatBufferKernel). Disabling GPU autotune keeps JIT training usable.
append_xla_flag('--xla_gpu_autotune_level=0')

# os.environ['JAX_LOG_COMPILES'] = '1'
# os.environ['JAX_COMPILATION_CACHE_DIR'] = os.environ.get('JAX_COMPILATION_CACHE_DIR', '/tmp/jax_cache')
# os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'
# os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.5.0'
os.environ['ROCM_PATH'] = '/opt/rocm'
os.environ['LLVM_PATH'] = '/opt/rocm/llvm'
if os.environ.get('LD_LIBRARY_PATH') is None:
    os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib'
else:
    os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH') + ':/opt/rocm/lib'

# --- 设置 Python 搜索路径 ---
# 获取当前脚本所在的目录 (src/jax)
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (NN)，即上两级目录
project_root = os.path.dirname(os.path.dirname(script_dir))
# 如果项目根目录不在搜索路径中，则添加它
if project_root not in sys.path:
    sys.path.append(project_root)

import jax
from jax import numpy as jnp
from layer import *

jax.config.update('jax_debug_nans', True)
jax.config.update('jax_debug_infs', True)

print("JAX version:", jax.__version__)

# # 简单测试以验证 JAX 是否正在使用 ROCm
cpu_device = jax.devices("cpu")[0]
devices = jax.devices()
try:
    # 检查是否有 GPU (ROCm) 设备
    gpu_device = [d for d in devices if d.platform == 'gpu'][0]
    print(f"JAX 正在使用 ROCm 后端: {gpu_device}")
except Exception as e:
    print("未检测到 ROCm 设备, JAX 可能正在使用 CPU 后端。")

_rng_key = jax.random.PRNGKey(0)

def seed_everything(seed=0):
    global _rng_key
    random.seed(seed)
    np.random.seed(seed)
    _rng_key = jax.random.PRNGKey(seed)

def myrandom():
    global _rng_key
    _rng_key, subkey = jax.random.split(_rng_key)
    return subkey

vocabulary = {
    'P' : 0,
    'S' : 1,
    'E' : 2,
    '1' : 3,
    '2' : 4,
    '3' : 5,
    '4' : 6,
    '5' : 7,
    '6' : 8,
    '7' : 9,
    '8' : 10,
    '9' : 11,
    '0' : 12,
    '+' : 13,
    '=' : 14
}
inv_vocabulary = {v: k for k, v in vocabulary.items()}

pad_token = vocabulary['P']

vocab_size = len(vocabulary)
d_model = 128
max_seqlen = 32
PE = positionalEncoding_init(max_seqlen, d_model)


def model_metadata():
    n_heads = 8
    return {
        'version': 2,
        'vocab_size': vocab_size,
        'd_model': d_model,
        'd_ff': 256,
        'd_k': d_model // n_heads,
        'd_v': d_model // n_heads,
        'n_heads': n_heads,
        'max_seqlen': max_seqlen,
    }


def build_model(meta):
    enc0 = encoderLayer_init(myrandom(), meta['d_model'], meta['d_ff'], meta['d_k'], meta['d_v'], meta['n_heads'])
    enc1 = encoderLayer_init(myrandom(), meta['d_model'], meta['d_ff'], meta['d_k'], meta['d_v'], meta['n_heads'])

    dec0 = decoderLayer_init(myrandom(), meta['d_model'], meta['d_ff'], meta['d_k'], meta['d_v'], meta['n_heads'])
    dec1 = decoderLayer_init(myrandom(), meta['d_model'], meta['d_ff'], meta['d_k'], meta['d_v'], meta['n_heads'])

    lin_o = tied_embedding_output_init(myrandom(), meta['d_model'], meta['vocab_size'])
    return (enc0, enc1, dec0, dec1, lin_o)

def transformer_init(drop_prob=0.0, lr=1e-4):
    meta = model_metadata()
    n_heads = meta['n_heads']

    # 检测模型文件是否存在
    if os.path.exists(file_to_save):
        print("检测到已保存的模型, 直接加载...")
        _, opt_configs = adamOpt_init(None, lr)
        with open(file_to_save, 'rb') as f:
            loaded_state = pickle.load(f)

        if loaded_state.get('meta') != meta:
            print("已保存模型结构与当前实现不兼容, 重新初始化参数...")
            model = build_model(meta)
            opt_state, opt_configs = adamOpt_init(model, lr)
            state = {
                'model': model,
                'opt_state': opt_state['opt_state'],
            }
        else:
            state = {
                'model': loaded_state['model'],
                'opt_state': loaded_state['opt_state'],
            }
    else:
        print("未检测到已保存的模型, 重新初始化模型参数...")
        model = build_model(meta)

        opt_state, opt_configs = adamOpt_init(model, lr)
        # 可训练的参数
        state = {
            'model': model,
            'opt_state': opt_state['opt_state'],
        }

    # 静态参数
    configs = {
        'n_heads' : n_heads,
        'drop_prob' : drop_prob,
        'meta': meta,
        **opt_configs
    }
    configs = StaticConfigs(configs)

    return state, configs

def transformer_apply(enc_ipt, dec_ipt, enc_padding_mask, lookahead_mask, params, n_heads, drop_prob, key=None):
    enc0, enc1, dec0, dec1, lin_o = params
    k1, k2, k3, k4 = split_key(key, 4)

    # encoder forwarding
    ini = embedding_apply(enc_ipt, lin_o[0].T) # 权重共享
    # ini *= jnp.sqrt(d_model)
    ini = positionalEncoding_encode(ini, PE)
    enc_output0 = encoderLayer_apply(ini, enc_padding_mask, enc0, n_heads, drop_prob, k1)
    enc_output = encoderLayer_apply(enc_output0, enc_padding_mask, enc1, n_heads, drop_prob, k2)

    # decoder forwarding
    ini = embedding_apply(dec_ipt, lin_o[0].T) # 权重共享
    # ini *= jnp.sqrt(d_model)
    ini = positionalEncoding_encode(ini, PE)
    dec_output0 = decoderLayer_apply(ini, enc_output, lookahead_mask, enc_padding_mask, dec0, n_heads, drop_prob, k3)
    dec_output = decoderLayer_apply(dec_output0, enc_output, lookahead_mask, enc_padding_mask, dec1, n_heads, drop_prob, k4)

    return linear_apply(dec_output, lin_o)

def transformerEncoder_apply(ipt, enc_padding_mask, params, n_heads, drop_prob, key=None):
    enc0, enc1, dec0, dec1, lin_o = params

    k1, k2 = split_key(key, 2)

    # encoder forwarding
    ini = embedding_apply(ipt, lin_o[0].T) # 权重共享
    # ini *= jnp.sqrt(d_model)
    ini = positionalEncoding_encode(ini, PE)
    enc_output0 = encoderLayer_apply(ini, enc_padding_mask, enc0, n_heads, drop_prob, k1)
    enc_output = encoderLayer_apply(enc_output0, enc_padding_mask, enc1, n_heads, drop_prob, k2)

    return enc_output

# @partial(jax.jit, static_argnames=('n_heads', 'drop_prob'), device=cpu_device)
def transformerDecoder_apply(ipt, enc_output, lookahead_mask, enc_padding_mask, params, n_heads, drop_prob, key=None):
    enc0, enc1, dec0, dec1, lin_o = params

    k1, k2 = split_key(key, 2)
    # decoder forwarding
    ini = embedding_apply(ipt, lin_o[0].T) # 权重共享
    # ini *= jnp.sqrt(d_model)
    ini = positionalEncoding_encode(ini, PE)
    dec_output0 = decoderLayer_apply(ini, enc_output, lookahead_mask, enc_padding_mask, dec0, n_heads, drop_prob, k1)
    dec_output = decoderLayer_apply(dec_output0, enc_output, lookahead_mask, enc_padding_mask, dec1, n_heads, drop_prob, k2)

    return linear_apply(dec_output, lin_o)


def _transformer_train(enc_indices, dec_indices, tgt_indices, params, n_heads, drop_prob, key):
    enc_padding_mask = padding_mask(enc_indices, pad_token)
    lookahead_mask = jnp.tril(jnp.ones((max_seqlen, max_seqlen)))[jnp.newaxis, :, :]
    dec_padding_mask = padding_mask(dec_indices, pad_token)
    # 组合解码器掩码
    combined_mask = jnp.minimum(lookahead_mask, dec_padding_mask)

    def compute_loss(params, key):
        logits = transformer_apply(enc_indices, dec_indices, enc_padding_mask, combined_mask, params, n_heads, drop_prob, key)
        loss = cross_entropy_loss_indices(tgt_indices, logits, ignore_index=pad_token)
        return loss

    # loss
    loss, grads = jax.value_and_grad(compute_loss)(params, key)
    return loss, grads

@partial(jax.jit, static_argnames=('configs',))
def transformer_train(enc_batch, dec_batch, tgt_batch, params, configs, key):
    batch_size = enc_batch.shape[0]

    model = params['model']
    opt_state = params['opt_state']

    n_heads = configs['n_heads']
    drop_prob = configs['drop_prob']
    opt_fn = configs['opt_fn']

    # 为批处理中的每个样本生成一个唯一的key
    keys = jax.random.split(key, batch_size)

    # vmap a single training step function
    # We want to average the losses and gradients across the batch
    losses, grads = jax.vmap(_transformer_train, in_axes=(0, 0, 0, None, None, None, 0))(
        enc_batch, dec_batch, tgt_batch, model, n_heads, drop_prob, keys
    )

    # Average gradients and loss across the batch
    avg_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)
    avg_loss = jnp.mean(losses)

    avg_grads = jax.tree_util.tree_map(
        lambda x: jnp.clip(x, -1.0, 1.0),
        avg_grads
    )

    # optimizer step
    updates, new_opt_state = opt_fn(avg_grads, opt_state, configs)
    new_model = jax.tree_util.tree_map(
        lambda p, u: p - u,
        model,
        updates
    )
    new_params = {
        'model': new_model,
        'opt_state': new_opt_state,
    }
    return new_params, avg_loss

def transformer_inference(ipt_indices, params, configs):
    model = params['model']
    n_heads = configs['n_heads']
    drop_prob = -1 # 推理时不使用 dropout

    # 1. Encoder Forwarding (只需计算一次)
    enc_padding_mask = padding_mask(ipt_indices, pad_token)
    enc_output = transformerEncoder_apply(ipt_indices, enc_padding_mask, model, n_heads, drop_prob)

    # 2. Auto-regressive Decoding
    # 使用 Python 列表来构建解码器输入
    dec_indices = [vocabulary['S']]
    
    for i in range(max_seqlen - 1):
        # 将当前解码序列转换为 JAX 数组并补齐
        current_seq_len = len(dec_indices)
        dec_input_arr = np.array(dec_indices)

        # 创建前瞻掩码
        lookahead_mask = np.tril(np.ones((1, current_seq_len, current_seq_len)))
        
        # 解码器前向传播
        dec_output = transformerDecoder_apply(dec_input_arr, enc_output, lookahead_mask, enc_padding_mask, model, n_heads, drop_prob)

        # 只关心最后一个时间步的输出
        predicted_id = np.argmax(dec_output[-1, :])
        
        # 如果预测到结束符，则停止
        if predicted_id == vocabulary['E']:
            break
        
        # 将预测结果添加到下一次的输入中
        dec_indices.append(int(predicted_id))

    # 将索引转换为字符，忽略第一个 'S'
    return "".join([inv_vocabulary[i] for i in dec_indices[1:]])


def string_to_indices(s):
    t = [vocabulary[c] for c in s]
    # 补齐到最大长度
    t = t + [vocabulary['P']] * (max_seqlen - len(t))
    return np.array(t)

def generate_dataset(batch_size, size=9999):
    # 生成训练集
    trainset = []
    for i in range(10000):
        a = random.randint(0, size)
        b = random.randint(0, size)
        ipt_str = f"{a}+{b}"
        tgt_str = f"{a}+{b}={a+b}"

        enc_ipt = 'S' + ipt_str + 'E'
        dec_ipt = 'S' + tgt_str
        dec_tgt = tgt_str + 'E'
        enc_indices = string_to_indices(enc_ipt)
        dec_indices = string_to_indices(dec_ipt)
        tgt_indices = string_to_indices(dec_tgt)
        trainset.append((enc_indices, dec_indices, tgt_indices))

    # split list
    batched_trainset = []
    for i in range(0, len(trainset), batch_size):
        batch = trainset[i:i + batch_size]
        if len(batch) == batch_size: # 确保是完整的batch
            enc_batch = np.array([item[0] for item in batch])
            dec_batch = np.array([item[1] for item in batch])
            tgt_batch = np.array([item[2] for item in batch])
            batched_trainset.append((enc_batch, dec_batch, tgt_batch))
    return batched_trainset

BATCH_SIZE = 128
file_to_save = 'transformer_jax_model.npz'

def main():
    seed_everything(0)
    train_state, train_configs = transformer_init()

    try:
        iter = 0
        size = 99
        for i in range(0, 120):
            total_loss = jnp.array(0.0)
            time_start = time.time()

            if i % 10 == 0 and i > 0:
                size *= 10
                size += 9
                print(f"增加训练数据量到 {size} ...")

            trainset = generate_dataset(BATCH_SIZE, size)
            shuffle_indices = list(range(len(trainset)))
            for idx in tqdm.tqdm(shuffle_indices, desc=f"{iter+1}:"):
                train_state, loss = transformer_train(
                    trainset[idx][0],
                    trainset[idx][1],
                    trainset[idx][2],
                    train_state,
                    train_configs,
                    myrandom()
                )
                total_loss += loss
            total_loss.block_until_ready()
            time_diff = time.time() - time_start
            print(f"耗时: {time_diff}, 平均损失: {total_loss / len(trainset)}")
            iter += 1

            if (iter) % 10 == 0:
                print("save model...")
                with open(file_to_save, 'wb') as f:
                    pickle.dump({
                        **train_state,
                        'meta': train_configs['meta'],
                    }, f)
    except KeyboardInterrupt:
        pass

    # 推理用cpu即可
    train_state = jax.device_put(train_state, device=cpu_device)
    train_state = jax.block_until_ready(train_state)
    with jax.default_device(cpu_device):
        print("训练结束, 进入测试阶段...")

        accuracy = 0
        for i in range(1000):
            a = random.randint(0, 9999)
            b = random.randint(0, 9999)
            ipt_str = f"{a}+{b}"
            tgt_str = f"{a}+{b}={a+b}"

            ipt_str = ipt_str[:max_seqlen - 2]
            ipt_indices = string_to_indices('S' + ipt_str + 'E')

            res = transformer_inference(ipt_indices, train_state, train_configs)
            print("output:", res)
            if res == tgt_str:
                accuracy += 1

        print(f"测试集准确率: {accuracy / 10}%")


if __name__ == "__main__":
    main()

# 人工测试
# with jax.default_device(cpu_device):
#     while True:
#         ipt_str = input("请输入加法表达式:")
#         if ipt_str == 'exit':
#             break
#         ipt_str = ipt_str[:max_seqlen - 2]
#         ipt_indices = string_to_indices('S' + ipt_str + 'E')
#         res = transformer_inference(ipt_indices, train_state, train_configs)
#         print("output:", res)
