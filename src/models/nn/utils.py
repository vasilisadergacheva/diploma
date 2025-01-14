import os
import torch
import shutil
from torch import nn
from typing import Any, Union, Callable, Optional
from torch.utils.data import random_split, Dataset, DataLoader, IterableDataset
from torch.utils.data._utils.collate import default_collate
from functools import wraps

from src.models.nn.trainers import TrainConfig


def random_split_dataset(
    dataset: Dataset,
    config: TrainConfig,
    to_loaders: bool = False,
    collate_fn: Callable = default_collate,
) -> tuple[Union[Dataset, DataLoader], Optional[Union[Dataset, DataLoader]]]:
    if config.valid_share is None:
        if to_loaders:
            return (
                DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    drop_last=True,
                    pin_memory=config.pin_memory,
                    collate_fn=collate_fn,
                ),
                None,
            )
        else:
            return dataset, None

    train_data, valid_data = random_split(
        dataset, [1 - config.valid_share, config.valid_share]
    )

    if to_loaders:
        train_data = DataLoader(
            train_data,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=config.pin_memory,
            collate_fn=collate_fn,
        )
        valid_data = DataLoader(
            valid_data,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=config.pin_memory,
            collate_fn=collate_fn,
        )

    return train_data, valid_data


def to_device(
    device: torch.device, tensors: Any, float32_to_float64: bool = True
) -> Any:
    """
    While for modules .to(device) is inplace, it is not correct for Tensors.
    """
    if isinstance(tensors, (list, tuple)):
        result = []
        for tensor in tensors:
            result.append(to_device(device, tensor))
        return result
    if tensors.dtype == torch.float64 and float32_to_float64:
        tensors = tensors.to(torch.float32)
    return tensors.to(device)


def is_dataset(obj: Any) -> bool:
    return hasattr(obj, "__getitem__") and hasattr(obj, "__len__")


def is_iterable_dataset(obj: Any) -> bool:
    return hasattr(
        obj,
        "__iter__",
    )


def pack(x, collate=False):
    if collate:
        x = default_collate([x])

    if isinstance(x, (list, tuple)):
        return x
    else:
        return (x,)


def eval_no_grad(func):
    """
    Decorator for both doing model.eval() and torch.no_grad()

    !model is the first argument!
            func(model, ...)
    """

    @wraps(func)
    def wrapper(model: nn.Module, *args, **kwargs):
        with torch.no_grad():
            model.eval()
            return func(model, *args, **kwargs)

    return wrapper


def clear_dir(path: str):
    shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)
