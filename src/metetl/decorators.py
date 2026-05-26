import time
from functools import wraps
import logging

logger = logging.getLogger("metetl")

def log_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug("Вызов %s", func.__name__)
        result = func(*args, **kwargs)
        logger.debug("Завершение %s", func.__name__)
        return result
    return wrapper

def time_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("Функция %s выполнена за %.2f сек", func.__name__, elapsed)
        return result
    return wrapper