import argparse

from benchs import get_benchmark
from common import default_logger as logger
from config import get_config, available_benchmark_names


def main():
    parser = argparse.ArgumentParser("mlp bench")
    parser.add_argument("--mode", type=str, default='precheck', choices=['dryrun', 'precheck'], help='mode')
    parser.add_argument("--driver-master-addr", type=str, default='ft-driver-master.mlp.svc.cluster.local:8080',
                        help='driver master addr')
    args = parser.parse_args()

    try:
        config = get_config(args.driver_master_addr).get('precheck', {})
    except Exception as e:
        logger.error(f"failed to get precheck config: {e}")
        raise e

    for benchmark_name in available_benchmark_names:
        benchmark_config = config.get(benchmark_name, {'enable': False})
        if not benchmark_config['enable']:
            logger.info(f'[{benchmark_name}] is disabled, skip')

        benchmark = get_benchmark(benchmark_name, **config.get(benchmark_name, {}))
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
