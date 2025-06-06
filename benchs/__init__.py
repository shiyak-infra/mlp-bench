from benchs.benchmark import *
from benchs.gpu_computation_precision import *
from benchs.gpu_flops import *
from benchs.nvlink import NVLink


def get_benchmark(name: str, **kwargs) -> Benchmark:
    if name == config.BENCHMARK_GPU_COMPUTATION_PRECISION:
        return GpuComputationPrecision(**kwargs)
    elif name == config.BENCHMARK_GPU_FLOPS:
        return GpuFlops(**kwargs)
    elif name == config.BENCHMARK_NVLINK:
        return NVLink(**kwargs)
    else:
        raise NotImplementedError()
