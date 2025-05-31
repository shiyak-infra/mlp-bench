import subprocess
import re

def run_cmd(cmd, cwd=None) -> (int, str, str):
    result = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode, result.stdout, result.stderr

def parse_stdout_to_dict(stdout: str) -> dict[str, float]:
    result = {}
    for line in stdout.strip().splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            try:
                result[key.strip()] = float(value.strip())
            except ValueError:
                raise RuntimeError(f"invalid output from {line}")
    return result

def get_gpu_product() -> str:
    ret_code, stdout, _ = run_cmd("nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -n 1")
    if ret_code != 0:
        raise RuntimeError(f"failed to get gpu product: {stdout}")
    return stdout.strip('\n').strip()