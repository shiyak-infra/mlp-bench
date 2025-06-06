from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import config
from benchs import benchmark, utils
from common import default_logger as logger


class NVLink(benchmark.Benchmark):
    def __init__(self, iterations: int = 100, warmup: int = 10, bandwidth: int = 330):
        super().__init__()
        self.iterations = iterations
        self.warmup = warmup
        self.bandwidth = bandwidth
        self.world_size = torch.cuda.device_count()
        self.matrix_size = utils._get_max_matrix_size(0, torch.float32)
        self.busbw = 0

    def name(self):
        return config.BENCHMARK_NVLINK

    def description(self):
        return "Stress test NVLink&NVSwitch bandwidth"

    def run(self):
        executor = ProcessPoolExecutor(max_workers=self.world_size)
        futures = []
        for rank in range(self.world_size):
            future = executor.submit(utils.bm_allreduce, rank, self.matrix_size, self.warmup, self.iterations, self.world_size, torch.float32)
            futures.append(future)

        for rank, future in enumerate(futures):
            if rank == 0:
                (_, algbw, busbw) = future.result()
                logger.info(f"algbw: {algbw:.3f}, busbw: {busbw:.3f}")
                if self.busbw < self.bandwidth:
                    return False, f"busbw {self.busbw} less than expected {self.bandwidth}"
        return True, ""



if __name__ == "__main__":
    benchmark = NVLink()
    ok, msg = benchmark.run()
    if not ok:
        print(msg)
