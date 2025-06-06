import argparse

import config
from benchs import get_benchmark
from common import default_logger as logger
from config import get_config, available_benchmark_names


def main():
    benchmark_name = config.BENCHMARK_NVLINK
    benchmark = get_benchmark(benchmark_name)
    logger.info(f"[{benchmark_name}] {benchmark.description()}")
    try:
        ok, msg = benchmark.run()
        if not ok:
            logger.error(f"[{benchmark_name}] benchmark failed: {msg}")
            raise RuntimeError(msg)
    except Exception as e:
        logger.error(f"[{benchmark_name}] caught exception: {e}")
        raise e


if __name__ == '__main__':
    main()
