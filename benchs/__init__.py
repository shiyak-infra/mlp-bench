from benchs.benchmark import *
from benchs.flop import *
from benchs.gpu_burn import *


def get_benchmark(name: str, **kwargs) -> Benchmark:
    if name == "gpu_burn":
        return GpuBurn(**kwargs)
    elif name == "flop":
        return Flop(**kwargs)
    else:
        raise NotImplementedError()
