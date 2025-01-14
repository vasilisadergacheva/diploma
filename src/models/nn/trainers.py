import torch
from typing import Optional
from dataclasses import dataclass

from src.models.nn.schedulers import Scheduler


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
    steps: Optional[int] = None

    batch_size: int = 8
    batches_accumulated: int = 1
    valid_share: Optional[float] = 0.2

    device: torch.device = torch.device("cuda:0")
    pin_memory: bool = False

    min_epoch_for_storing: int = 1


class Trainer:
    def reset(self) -> None:
        """
        Backpropagation.
        """
        raise NotImplementedError

    def step(self, predicted, target) -> float:
        """
        Returns step loss.
        """
        raise NotImplementedError

    def get_loss(self, predicted, target) -> float:
        """
        Just returns loss.
        """
        raise NotImplementedError

    def end_epoch(self) -> None:
        """
        Update internal state after epoch.
        """
        raise NotImplementedError


class DefaultTrainer(Trainer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        config: TrainConfig,
        scheduler: Optional[Scheduler] = None,
    ):
        """
        Default trainer with the ability to accumulate batches.
        """

        self.optimizer = optimizer
        self.optimizer.zero_grad()

        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.steps_accumulated = config.batches_accumulated
        self.batch_size: int = config.batch_size
        self.max_step: Optional[int] = config.steps
        self.current_step: int = 0

        if self.steps_accumulated == 1:
            assert loss_fn.reduction == "mean"
            self.factor = 1
        else:
            assert loss_fn.reduction == "sum"
            self.factor = 1 / (self.steps_accumulated * self.batch_size)

    def reset(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.scheduler is not None:
            self.scheduler.step_after_optimizer()

    def step(self, predicted, target) -> float:
        self.current_step += 1

        if (
            self.max_step is not None
            and self.current_step > self.max_step * self.steps_accumulated
        ):
            raise StopIteration

        loss = self.loss_fn(predicted, target) * self.factor
        loss.backward()

        if self.current_step % self.steps_accumulated == 0:
            self.reset()

        if self.steps_accumulated == 1:
            return loss.item()
        else:
            return loss.item() * self.steps_accumulated

    @torch.no_grad()
    def get_loss(self, predicted, target) -> float:
        loss = self.loss_fn(predicted, target)

        if self.steps_accumulated == 1:
            return loss.item()
        else:
            return loss.item() / self.batch_size

    def end_epoch(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step_after_epoch()
