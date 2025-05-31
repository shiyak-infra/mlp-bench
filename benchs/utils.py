import functools
import subprocess

import torch


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


def disable_tf32(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        orig_matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        orig_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            return func(*args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = orig_matmul_allow_tf32
            torch.backends.cudnn.allow_tf32 = orig_cudnn_allow_tf32

    return wrapper
