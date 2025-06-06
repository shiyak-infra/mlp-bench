import torch
from concurrent.futures import ProcessPoolExecutor

import config
from benchs import benchmark, utils, flops_utils
from common import default_logger as logger

from multiprocessing import Process


class GpuFlops(benchmark.Benchmark):
    def __init__(self, iterations: int = 100, warmup: int = 20, threshold: float = 0.6,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.iterations = iterations
        self.warmup = warmup
        self.threshold = threshold
        self.device_count = torch.cuda.device_count()
        self.dtype = dtype

    def name(self):
        return config.BENCHMARK_GPU_FLOPS

    def description(self):
        return "Execute pytorch's matmul to stress test whether the device can reach the theoretical flops"

    def run(self):
        executor = ProcessPoolExecutor(max_workers=self.device_count)
        futures = []
        for rank in range(self.device_count):
            future = executor.submit(utils.matmul, rank, self.warmup, self.iterations, self.dtype)
            futures.append(future)

        for rank, future in enumerate(futures):
            (_, tflops) = future.result()
            available_flops = flops_utils.get_available_flops(torch.device(f"cuda:{rank}"), self.dtype)
            if available_flops is None:
                logger.warning("device available flops not found, set to tflops")
                available_flops = tflops * 1e12
            available_tflops = available_flops / 1e12
            logger.info(
                f"device:{rank} dtype:{str(self.dtype)} mfu:{tflops / available_tflops * 100:.3f}% tflops:{tflops:.3f}, available_tflops:{available_tflops}")
            if tflops < available_tflops * self.threshold:
                return False, f"gpu {rank} dtype {str(self.dtype)} tflops {tflops:.3f} less than available {available_tflops} with threshold {self.threshold}"
        return True, ""


if __name__ == "__main__":
    benchmark = GpuFlops()
    ok, msg = benchmark.run()
    if not ok:
        print(msg)
