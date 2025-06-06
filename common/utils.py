import os

import torch

from common import default_logger as logger


LOCAL_RANK = "LOCAL_RANK"


def log_execution_time(func):
    def wrapper(*args, **kwargs):
        local_rank = int(os.environ[LOCAL_RANK])
        func_name = func.__name__
        logger.info(f"Begin execute {func_name} on local rank {local_rank}")
        t = func(*args, **kwargs)
        t = round(t, 3)
        logger.info(
            f"Time to execute {func_name} on local rank {local_rank} is {t}s."
        )
        return t

    return wrapper


def disable_tf32(func):
    def wrapper(*args, **kwargs):
        orig_matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        orig_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
        try:
            logger.info("disable tf32 support")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            return func(*args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = orig_matmul_allow_tf32
            torch.backends.cudnn.allow_tf32 = orig_cudnn_allow_tf32

    return wrapper
