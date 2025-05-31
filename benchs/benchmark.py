import abc
import typing


class Benchmark(abc.ABC):
    def __init__(self):
        return

    def name(self) -> str:
        raise NotImplementedError

    def run(self) -> (bool, str):
        raise NotImplementedError
