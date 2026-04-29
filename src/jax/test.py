import os

use_cpu = False

if use_cpu:
    os.environ['JAX_PLATFORMS'] = 'cpu'

# os.environ['JAX_LOG_COMPILES'] = '1'
os.environ['JAX_COMPILATION_CACHE_DIR'] = os.path.expanduser('~/.jax_cache')
os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.5.1'
os.environ['ROCM_PATH'] = '/opt/rocm'
os.environ['LLVM_PATH'] = '/opt/rocm/llvm'
if os.environ.get('LD_LIBRARY_PATH') is None:
    os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib'
else:
    os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH') + ':/opt/rocm/lib'

import jax
# import jax.numpy as jnp
# from jax import random
import time

print(f"JAX 版本: {jax.__version__}")

def main():
    # 检查 JAX 是否能检测到 GPU
    try:
        print("1")
        print(f"JAX 检测到 {len(jax.devices())} 个设备: {jax.devices()}")
        print("2")
        if any('gpu' in device.platform.lower() for device in jax.devices()):
            print("GPU 已成功被 JAX 检测到。")
        else:
            print("警告：JAX 未检测到 GPU。代码将在 CPU 上运行。")
    except Exception as e:
        print(f"检查 JAX 设备时出错: {e}")

    return

    # 使用一个 key 来生成伪随机数，这是 JAX 的要求
    key = random.PRNGKey(0)

    # 创建两个大的随机矩阵
    size = 256
    key, subkey1, subkey2 = random.split(key, 3)
    mat1 = random.normal(subkey1, (size, size))
    mat2 = random.normal(subkey2, (size, size))

    # 定义一个简单的矩阵乘法函数
    def multiply(a, b):
        return jnp.dot(a, b)

    # 使用 @jit 装饰器来编译这个函数以便在 GPU 上加速
    fast_multiply = jax.jit(multiply)

    # --- 第一次运行 (包含编译时间) ---
    print(f"\n正在对 {size}x{size} 的矩阵进行第一次 JIT 编译和计算...")
    start_time = time.time()
    # 执行 JIT 编译的函数
    result1 = fast_multiply(mat1, mat2)
    # JAX 的操作是异步的。我们需要调用 .block_until_ready() 来等待计算完成并获得准确的执行时间
    result1.block_until_ready()
    duration1 = time.time() - start_time
    print(f"第一次运行 (含编译) 耗时: {duration1:.4f} 秒")

    # --- 第二次运行 (使用已编译的缓存) ---
    print(f"\n正在进行第二次计算 (使用已编译的函数)...")
    start_time = time.time()
    result2 = fast_multiply(mat1, mat2)
    result2.block_until_ready()
    duration2 = time.time() - start_time
    print(f"第二次运行耗时: {duration2:.4f} 秒")

    print(f"\n第一次运行比第二次运行慢了约 {(duration1 - duration2):.4f} 秒，这部分主要是 JIT 的编译开销。")


if __name__ == "__main__":
    main()