import sys


if "IPython" in sys.modules:
    from tqdm.notebook import tqdm  # type: ignore
else:
    from tqdm import tqdm  # type: ignore

from torch.utils.tensorboard.writer import SummaryWriter

from typing import Optional
from dataclasses import dataclass


@dataclass
class LoggerConfig:
    # common
    log_interval: int = 1
    step_bias: int = 0
    reduction: str = "last"
    # tensorboard
    tb_logs: bool = False
    log_dir: str = "logs"
    group: Optional[str] = None
    # tqdm
    text_logs: bool = False
    title: tuple[str, str] = ("iteration", "value")
    tqdm_total: Optional[int] = None
    tqdm_position: Optional[int] = None


@dataclass(frozen=True)
class TrainLossLoggerConfig:
    log_dir: str = "logs"
    text_logs: bool = False
    tb_logs: bool = False
    step_logs: bool = False
    log_interval: int = 1


class Logger:
    def __init__(self, config: LoggerConfig):
        self.config = config

        self.current_step: int = 0
        self.current_values: list[float] = []
        self.tb_writer = SummaryWriter(log_dir=self.config.log_dir)
        self.pbar = (
            None
            if not self.config.text_logs
            else tqdm(
                total=self.config.tqdm_total,
                position=self.config.tqdm_position,
                desc=self.config.title[0],
            )
        )

        def _set_text_log(iteration, value):
            if self.config.text_logs:
                self.pbar.update(iteration - self.pbar.n)  # type: ignore
                self.pbar.set_postfix({self.config.title[1]: value})  # type: ignore

        def _set_tb_log(iteration, value):
            if self.config.tb_logs:
                self.tb_writer.add_scalar(self.config.group, value, iteration)

        self._set_text_log = _set_text_log
        self._set_tb_log = _set_tb_log

    def reset(self) -> None:
        match self.config.reduction:
            case "last":
                value = self.current_values[-1]
            case "sum":
                value = sum(self.current_values)
            case "mean":
                value = sum(self.current_values) / len(self.current_values)
            case _:
                raise NotImplementedError

        self._set_tb_log(self.current_step + self.config.step_bias, value)
        self._set_text_log(self.current_step + self.config.step_bias, value)
        self.current_values = []
        self.tb_writer.flush()

    def step(self, value: float) -> float:
        """
        Returns given value.
        """

        self.current_step += 1
        self.current_values.append(value)

        if self.current_step % self.config.log_interval == 0:
            self.reset()

        return value

    def add_step_bias(self, value: int) -> None:
        self.config.step_bias += value

    def __del__(self):
        self.tb_writer.close()
        if not (self.pbar is None):
            self.pbar.close()
