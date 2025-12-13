"""
Author: zzc0721@github
"""

from loguru import logger
import torch
import time
import sys


def benchmark_precision(precision, matrix_size, warmup=6, test_iters=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError("需要CUDA显卡进行测试")

    # 初始化矩阵
    try:
        a = torch.randn(matrix_size, matrix_size, dtype=precision, device=device)
        b = torch.randn(matrix_size, matrix_size, dtype=precision, device=device)
    except RuntimeError as e:
        if "not implemented" in str(e):
            return None
        raise

    # 预热
    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()

    # 正式测试
    start_time = time.time()
    for _ in range(test_iters):
        torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    # 计算FLOPS
    flops_per_iter = 2 * matrix_size**3
    total_flops = flops_per_iter * test_iters
    tflops = (total_flops / elapsed) / 1e12

    # 清理显存
    del a, b
    torch.cuda.empty_cache()

    return tflops


def main():
    yield "本 BenchMark 脚本来自 zzc0721@github，仓库：https://github.com/zzc0721/torch-performance-test-data"
    if not torch.cuda.is_available():
        yield ("未检测到CUDA设备")
        return

    device_name = torch.cuda.get_device_name(0)
    yield (f"测试设备: {device_name}")
    yield (
        f"显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n"
    )
    yield (f"Python版本: {sys.version}")
    yield (f"PyTorch版本: {torch.__version__}")

    # 测试不同精度和矩阵大小
    matrix_sizes = [1024, 2048, 4096, 8192, 10240]
    precisions = [
        ("FP32", torch.float32),
        ("FP16", torch.float16),
        ("BF16", torch.bfloat16),
    ]

    results = {}
    for precision_name, precision in precisions:
        yield (f"\n测试 {precision_name}:")
        results[precision_name] = []

        for size in matrix_sizes:
            yield (f"测试矩阵大小: {size}x{size}")
            tflops = benchmark_precision(precision, size)
            if tflops is not None:
                results[precision_name].append((size, tflops))
                yield (f"  性能: {tflops:.2f} TFLOPS")
            else:
                yield ("  不支持此精度")
                break

    # 打印总结
    yield ("\n性能总结:")
    for precision_name, measurements in results.items():
        if measurements:
            max_tflops = max(tflops for _, tflops in measurements)
            yield (f"{precision_name} 最大算力: {max_tflops:.2f} TFLOPS")


def benchmark_wrapper():
    try:
        result = ""
        for message in main():
            result += message + "\n"
            logger.info(message)
            yield result

    except Exception as e:
        logger.error(f"Benchmark 错误: {e}")
