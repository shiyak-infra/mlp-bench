import logging
import typing

import torch

from benchs import benchmark
from benchs import utils

logger = logging.getLogger(__name__)

# TODO(zuti.he) 根据NV的数据给出基准
_GPU_BURN_FLOPS: dict[str, dict[typing.Union[str, torch.dtype], float]] = {
    "NVIDIA H100 80GB HBM3": {
        torch.float64: 53e12,
        torch.float32: 51e12,
    }
}


class GpuBurn(benchmark.Benchmark):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def name(self):
        return "gpu_burn"

    def run(self):
        # NOTE(zuti.he) 这里只考虑了8卡
        ret_code, stdout, _ = utils.run_cmd("gpu_burn -m 100% -stts 10 60 2>&1 | tail -n 8")
        if ret_code != 0:
            raise RuntimeError("gpu_burn failed")
        result = utils.parse_stdout_to_dict(stdout)

        gpu_product = utils.get_gpu_product()
        if gpu_product not in _GPU_BURN_FLOPS[gpu_product]:
            logger.warning(f"gpu product {gpu_product} not found in flops dict, skip it")
            return True, ""
        if torch.float32 not in _GPU_BURN_FLOPS[gpu_product]:
            logger.warning(f"torch.float32 not found in flops dict, skip it")
            return True, ""
        gpu_product_flops = _GPU_BURN_FLOPS[gpu_product][torch.float32]
        expect = gpu_product_flops / 10e12

        for gpu, gflops in result.items():
            have = gflops / 10e3
            if have < expect * self.threshold:
                return False, f"gpu {gpu} tflops {have} less than expected {expect} with threshold {self.threshold}"
        return True, ""
