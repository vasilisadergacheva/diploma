from typing import Callable, Any


def apply(_dict: dict[Any, Any], func: Callable) -> dict:
    return dict(zip(_dict, map(func, _dict.values())))

