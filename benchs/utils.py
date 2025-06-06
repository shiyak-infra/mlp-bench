import math
import os
import subprocess
from datetime import timedelta

import pynvml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import common
from common import default_logger as logger
from benchs import flops_utils

MAX_MATRIX_SIZE = 16384  # 手动选择的矩阵大小, 实际上不一定是最大flop的矩阵参数


def run_cmd(cmd, cwd=None) -> (int, str, str):
    result = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode, result.stdout, result.stderr


def parse_stdout_to_dict(stdout: str) -> dict[str, str]:
    """
    Input:
    test1: value1
    test2: value2
    Output:
    {"test1": "value1", "test2": "value2"}
    """
    result = {}
    for line in stdout.strip().splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            try:
                result[key.strip()] = value.strip()
            except ValueError:
                raise RuntimeError(f"invalid output from {line}")
    return result


def matmul(rank: int, warmup: int, round_num: int, dtype: torch.dtype) -> (float, float):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    matrix_size = _get_matrix_size(rank, dtype)
    logger.info(f"using matrix size {matrix_size}")

    a = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
    c = torch.empty(matrix_size, matrix_size, device=device, dtype=dtype)

    for _ in range(warmup):
        torch.matmul(a, b, out=c)

    # real benchmark
    elapsed = _execute_gpu_matmul(a, b, round_num, c)
    tflops = 2 * matrix_size ** 3 / elapsed / 1e12
    return elapsed, tflops


@common.disable_tf32
@common.log_execution_time
def _execute_gpu_matmul(tensor1: torch.Tensor, tensor2: torch.Tensor, round_num: int,
                        out: torch.Tensor = None) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream=torch.cuda.current_stream())

    for _ in range(round_num):
        with torch.no_grad():
            torch.matmul(tensor1, tensor2, out=out)

    end.record(stream=torch.cuda.current_stream())
    end.synchronize()
    elapsed_time = start.elapsed_time(end) / 1000 / round_num
    return elapsed_time


def _get_matrix_size(rank: int, dtype: torch.dtype) -> int:
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mem = meminfo.free
        matrix_size = MAX_MATRIX_SIZE
        while 3 * matrix_size ** 2 * torch.tensor([], dtype=dtype).element_size() > free_mem:
            matrix_size /= 2  # 如果可用显存小于需要分配的矩阵大小, 除以2
        return matrix_size
    finally:
        pynvml.nvmlShutdown()


def _get_max_matrix_size(device_id: int, dtype: torch.dtype) -> int:
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mem = meminfo.free
        matrix_size = math.ceil(math.sqrt(free_mem / torch.tensor([], dtype=dtype).element_size()))
        return matrix_size

    finally:
        pynvml.nvmlShutdown()


def bm_allreduce(rank: int, shape: int, warmup: int, iterations: int, world_size:int, dtype: torch.dtype) -> (float, float, float):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    os.environ[common.LOCAL_RANK] = f'{rank}'
    dist.init_process_group("nccl", timeout=timedelta(seconds=10), rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")
    data = torch.randn(shape, shape, dtype=dtype).to(device)

    elapsed_time = _execute_nccl_comm(dist.all_reduce, rank, warmup, iterations, data)
    elapsed_time = 0.001 if elapsed_time == 0.0 else elapsed_time
    dist.destroy_process_group()

    gb_unit = 1024 * 1024 * 1024
    algobw = shape * 4 / gb_unit / (elapsed_time / 1000)
    busbw = algobw * 2 * (world_size - 1) / world_size
    algobw = round(algobw, 2)
    busbw = round(busbw, 2)
    elapsed_time = round(elapsed_time, 3)
    return elapsed_time, algobw, busbw


@common.log_execution_time
def _execute_nccl_comm(comm_op, device_id: int, warmup: int, iterations: int, *args) -> float:
    # warm up
    for _ in range(warmup):
        comm_op(*args)
    torch.cuda.synchronize(device=device_id)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream=torch.cuda.current_stream())

    for _ in range(iterations):
        comm_op(*args)

    end.record(stream=torch.cuda.current_stream())
    end.synchronize()
    elapsed_time = start.elapsed_time(end) / iterations
    return elapsed_time
