import time
from multiprocessing import Process, Queue

import pynvml
import torch

from benchs import benchmark, cuda_flop, utils

MAX_MATRIX_SIZE = 8192  # 手动选择的矩阵大小, 实际上不一定是最大flop的矩阵参数


class Flop(benchmark.Benchmark):
    def __init__(self, iterations: int = 100, warmup: int = 50, threshold: float = 0.8,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.iterations = iterations
        self.warmup = warmup
        self.threshold = threshold
        self.dtype = dtype
        self.device_count = torch.cuda.device_count()

    def name(self):
        return "flop"

    def description(self):
        return "Execute pytorch's matmul to stress test whether the device can reach the theoretical flops"

    @utils.disable_tf32
    def run(self):
        processes = []
        queue = Queue()
        for device_id in range(self.device_count):
            p = Process(target=self._run_device, args=(device_id, queue))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        for _ in range(len(processes)):
            (device_id, tflops) = queue.get()
            available_flops = cuda_flop.get_available_flops(torch.device(f'cuda:{device_id}'), self.dtype)
            if available_flops is None:
                self.log("device available flops not found, set to tflops")
                available_flops = tflops * 10e12
            available_tflops = available_flops / 10e12
            self.log(
                f"device:{device_id} dtype:{str(self.dtype)} mfu:{tflops / available_tflops * 100}% tflops:{tflops}, available_tflops:{available_tflops}")
            if tflops < available_tflops * self.threshold:
                return False, f"gpu {device_id} dtype {str(self.dtype)} tflops {tflops} less than available {available_tflops} with threshold {self.threshold}"
        return True, ""

    def _run_device(self, device_id: int, queue: Queue):
        device = torch.device(f'cuda:{device_id}')
        matrix_size = self._get_matrix_size(device_id)
        self.log(f"using matrix size {matrix_size}")

        a = torch.randn(matrix_size, matrix_size, device=device, dtype=self.dtype)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=self.dtype)
        c = torch.empty(matrix_size, matrix_size, device=device, dtype=self.dtype)

        # warmup
        for _ in range(100):
            torch.matmul(a, b, out=c)

        # real benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(self.iterations):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize()
        end = time.time()
        elapsed = end - start
        tflops = 2 * matrix_size ** 3 * self.iterations / elapsed / 1e12

        queue.put((device_id, tflops))

    def _get_matrix_size(self, device_id: int) -> int:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mem = meminfo.free
        pynvml.nvmlShutdown()

        matrix_size = MAX_MATRIX_SIZE
        while 3 * matrix_size ** 2 * torch.tensor([], dtype=self.dtype).element_size() > free_mem:
            matrix_size /= 2  # 如果可用显存小于需要分配的矩阵大小, 除以2
        return matrix_size
