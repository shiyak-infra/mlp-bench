import argparse
import json

from benchs import get_benchmark


def main():
    parser = argparse.ArgumentParser("mlp bench")
    parser.add_argument("--mode", type=str, default='precheck', choices=['dryrun', 'precheck'], help='mode')
    parser.add_argument("--benchmarks", type=str, default='gpu_burn,flop', help='benchmark list, split by comma')
    parser.add_argument("--config", type=str, default="./config.json", help='path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    for benchmark_name in args.benchmarks.split(","):
        if benchmark_name == "":
            continue
        if args.mode == 'dryrun':
            print(f"[{benchmark_name}] dryrun pass")
            continue
        benchmark = get_benchmark(benchmark_name, **config.get(benchmark_name, {}))
        print(f"[{benchmark.name()}] {benchmark.description()}")
        try:
            ok, msg = benchmark.run()
            if not ok:
                print(f"[{benchmark.name()}] benchmark failed: {msg}")
                raise RuntimeError(msg)
        except Exception as e:
            print(f"[{benchmark.name()}] caught exception: {e}")
            raise e


if __name__ == '__main__':
    main()
