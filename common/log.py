import logging
import sys
import typing

logging.basicConfig(level=logging.INFO)

_DEFAULT_LOGGER = "precheck.logger"

_LOGGER_LEVEL_RANGE = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_DEFAULT_FORMATTER = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] "
    "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
)

_ch = logging.StreamHandler(stream=sys.stderr)
_ch.setFormatter(_DEFAULT_FORMATTER)

_DEFAULT_HANDLERS = [_ch]

_LOGGER_CACHE: typing.Dict[str, logging.Logger] = {}


def get_logger(name, level: str = "INFO", handlers=None, update: bool = False):
    if name in _LOGGER_CACHE and not update:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = handlers or _DEFAULT_HANDLERS
    logger.propagate = False
    return logger


default_logger = get_logger(_DEFAULT_LOGGER)
