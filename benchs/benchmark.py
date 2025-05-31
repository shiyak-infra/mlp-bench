import abc


class Benchmark(abc.ABC):
    def __init__(self):
        return

    def name(self) -> str:
        raise NotImplementedError()

    def description(self) -> str:
        raise NotImplementedError()

    def run(self) -> (bool, str):
        raise NotImplementedError()

    def log(self, msg: str):
        print(f'[{self.name()}] {msg}')
