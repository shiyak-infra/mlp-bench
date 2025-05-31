import torch

from benchs import utils, benchmark


class GpuBurn(benchmark.Benchmark):
    def __init__(self, cwd: str = '/gpu-burn', duration: int = 30):
        super().__init__()
        self.cwd = cwd
        self.duration = duration
        self.device_count = torch.cuda.device_count()

    def name(self):
        return "gpu_burn"

    def description(self):
        return "Execute gpu_burn to verify the accuracy and availability of GPU"

    def run(self):
        ret_code, stdout, stderr = utils.run_cmd(
            f"./gpu_burn -m 100% -stts 10 {self.duration} 2>&1 | tail -n {self.device_count}", cwd=self.cwd)
        if ret_code != 0:
            return False, f"failed to run gpu_burn, return code: {ret_code}, stdout: {stdout}, stderr: {stderr}"
        result = utils.parse_stdout_to_dict(stdout)
        if len(result) != self.device_count:
            return False, f"invalid gpu_burn result, stdout: {stdout}, stderr: {stderr}"
        self.log(f"result:{result}")
        for k, v in result.items():
            if v != "OK":
                return False, f"gpu {k} failed to pass gpu_burn {v}"
        return True, ""
