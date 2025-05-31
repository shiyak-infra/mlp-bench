from benchs import benchmark


class AllReduce(benchmark.Benchmark):
    def __init__(self, bandwidth: int = 330):
        super().__init__()
        self.bandwidth = bandwidth

    def name(self):
        return "all_reduce"

    def description(self):
        return "Stress rdma bandwidth"

    def run(self):
        raise NotImplementedError()

    def _run_deepspeed(self) -> (bool, str):
        raise NotImplementedError()

    def _run_pytorch(self) -> (bool, str):
        raise NotImplementedError()
