import time

import torch
import torch.nn.functional as F


def log(message: str) -> None:
    print(message, flush=True)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        log("  before synchronize")
        torch.cuda.synchronize(device)
        log("  after synchronize")


def bench(name: str, fn, device: torch.device):
    log(f"{name}: start")
    sync(device)
    start = time.perf_counter()
    out = fn()
    sync(device)
    elapsed = time.perf_counter() - start
    log(f"{name}: {elapsed:.4f}s")
    return out


def main():
    log(f"torch: {torch.__version__}")
    log(f"hip: {getattr(torch.version, 'hip', None)}")
    log(f"cuda_available: {torch.cuda.is_available()}")
    log(f"device_count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(idx)
            log(
                f"device[{idx}]: name={prop.name}, "
                f"total_memory={prop.total_memory / 1024**3:.2f} GiB, "
                f"multi_processor_count={prop.multi_processor_count}"
            )
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        log("warning: no ROCm GPU visible, running CPU fallback only")

    log(f"selected_device: {device}")

    with torch.no_grad():
        log("alloc_empty: start")
        empty = torch.empty((1,), device=device, dtype=torch.float32)
        log(f"alloc_empty: ok shape={tuple(empty.shape)}")

        log("alloc_zeros: start")
        zeros = torch.zeros((1,), device=device, dtype=torch.float32)
        sync(device)
        log(f"alloc_zeros: ok shape={tuple(zeros.shape)}")

        log("host_to_device_copy: start")
        copied = torch.ones((1,), dtype=torch.float32).to(device)
        sync(device)
        log(f"host_to_device_copy: ok shape={tuple(copied.shape)}")

        a = torch.randn((1024, 1024), device=device, dtype=torch.float32)
        b = torch.randn((1024, 1024), device=device, dtype=torch.float32)
        c = bench("matmul", lambda: a @ b, device)
        log(f"matmul_result: shape={tuple(c.shape)} mean={c.mean().item():.6f}")

        x = torch.randn((8, 3, 224, 224), device=device, dtype=torch.float32)
        w = torch.randn((32, 3, 7, 7), device=device, dtype=torch.float32)
        y = bench(
            "conv2d",
            lambda: F.conv2d(x, w, stride=2, padding=3),
            device,
        )
        log(f"conv2d_result: shape={tuple(y.shape)} std={y.std().item():.6f}")

        v = torch.linspace(0, 1, steps=1_000_000, device=device, dtype=torch.float32)
        s = bench("elementwise", lambda: (v.sin() * v.cos()).sum(), device)
        log(f"elementwise_result: {s.item():.6f}")

        if device.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            log(
                "memory_info: "
                f"free={free_bytes / 1024**3:.2f} GiB, "
                f"total={total_bytes / 1024**3:.2f} GiB"
            )

    log("done")


if __name__ == "__main__":
    main()
