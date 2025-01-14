from typing import Optional, Union
import torch
import weakref
import warnings
from functools import wraps

from torch.optim.lr_scheduler import LRScheduler

from src.models.nn.logger import Logger, LoggerConfig


class Scheduler:
    """
    len(loggers) == len(param_groups)
    """

    loggers: Optional[list[Logger]] = None

    def step_after_optimizer(self) -> None:
        """
        Step after gradient step.
        """
        raise NotImplementedError

    def step_after_epoch(self) -> None:
        """
        Step after epoch.
        """
        raise NotImplementedError

    def current_lr(self) -> list[float]:
        """
        Just check current lr. Must not to change it.
        """
        raise NotImplementedError

    def track_lr_wrapper(method):  # type: ignore
        if getattr(method, "_track_lr_wrapper", False):
            # method has already been replaced, return.
            return method

        # class instance contains in __self__
        instance_ref = weakref.ref(method.__self__)  # type: ignore

        # method is wrapper over __func__
        func = method.__func__  # type: ignore

        # cls
        cls = instance_ref().__class__

        # del previous method
        del method

        @wraps(func)
        def wrapper(*args, **kwargs):
            instance = instance_ref()

            if instance.loggers is not None:  # type: ignore
                current_lrs = instance.current_lr()  # type: ignore

                assert len(instance.loggers) == len(current_lrs)  # type: ignore
                for i in range(len(instance.loggers)):  # type: ignore
                    instance.loggers[i].step(current_lrs[i])  # type: ignore

            wrapped = func.__get__(instance, cls)
            return wrapped(*args, **kwargs)

        wrapper._track_lr_wrapper = True  # type: ignore
        return wrapper


class DefaultScheduler(Scheduler):
    def __init__(
        self,
        logger_configs: Union[list[LoggerConfig], LoggerConfig],
        scheduler: LRScheduler,
        after_epoch=True,
    ):
        self.scheduler = scheduler
        self.after_epoch = after_epoch

        self.loggers = []
        for i in range(len(self.scheduler.optimizer.param_groups)):
            self.loggers.append(
                Logger(
                    (
                        logger_configs[i]
                        if isinstance(logger_configs, list)
                        else logger_configs
                    ),
                )
            )

        self.step_after_optimizer = Scheduler.track_lr_wrapper(
            self.step_after_optimizer  # type: ignore
        )
        self.step_after_epoch = Scheduler.track_lr_wrapper(
            self.step_after_epoch  # type: ignore
        )

    def step_after_optimizer(self):
        if not self.after_epoch:
            self.scheduler.step()

    def step_after_epoch(self):
        if self.after_epoch:
            self.scheduler.step()

    def current_lr(self):
        return [group["lr"] for group in self.scheduler.optimizer.param_groups]


class STLRScheduler(LRScheduler):
    """
    Slanted Triangular Learning Rate.
    When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_epoch: int,
        warming_epochs: int,
        peak_lr: list[float],
        last_epoch=-1,
        verbose="deprecated",
    ):
        assert warming_epochs > peak_epoch

        self.peak_epoch = peak_epoch
        self.last_epoch = last_epoch
        self.warming_epochs = warming_epochs
        self.peak_lr = peak_lr
        super().__init__(optimizer, last_epoch, verbose)

        self.lambda_before_peak = [
            (peak - base_lr) / peak_epoch
            for peak, base_lr in zip(self.peak_lr, self.base_lrs)
        ]
        self.lambda_after_peak = [
            (base_lr - peak) / (warming_epochs - peak_epoch)
            for peak, base_lr in zip(self.peak_lr, self.base_lrs)
        ]

    def get_lr(self):
        if not self._get_lr_called_within_step:  # noqa from LRScheduler
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0 or self.last_epoch > self.warming_epochs:
            return [group["lr"] for group in self.optimizer.param_groups]
        else:
            return [
                group["lr"] + lamb
                for group, lamb in zip(
                    self.optimizer.param_groups,
                    (
                        self.lambda_before_peak
                        if self.last_epoch <= self.peak_epoch
                        else self.lambda_after_peak
                    ),
                )
            ]

    def _get_closed_form_lr(self):
        if self.last_epoch > self.warming_epochs:
            return self.base_lrs
        elif self.last_epoch <= self.peak_epoch:
            return [
                base_lr + self.last_epoch * lamb
                for lamb, base_lr in zip(self.lambda_before_peak, self.base_lrs)
            ]
        else:
            return [
                peak_lr - (self.last_epoch - self.peak_epoch) * lamb
                for lamb, peak_lr in zip(self.lambda_after_peak, self.peak_lr)
            ]
