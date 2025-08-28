"""Utility helpers including timer and exponential retry."""

import functools
import time
from contextlib import contextmanager
from typing import Callable, Tuple, Type, TypeVar

from .logging import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


@contextmanager
def timer(name: str):
    """Context manager measuring execution time."""

    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.info(f"{name} took {duration:.3f}s")


def retry(
    exceptions: Tuple[Type[BaseException], ...],
    tries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry decorator with exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    logger.warning(
                        "retry", extra={"exc": str(exc), "sleep": _delay, "tries": _tries}
                    )
                    time.sleep(_delay)
                    _tries -= 1
                    _delay *= backoff
            return func(*args, **kwargs)

        return wrapper

    return decorator
