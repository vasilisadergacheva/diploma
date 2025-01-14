from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.nn.trainers import Trainer, TrainConfig
from src.models.nn.logger import Logger, LoggerConfig, TrainLossLoggerConfig
from src.models.nn.utils import (
    pack,
    to_device,
    eval_no_grad,
    random_split_dataset,
    clear_dir,
)
from src.metrics import DynamicMetric


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    trainer: Trainer,
    logger: Logger,
) -> float:
    """
    Returns average loss.
    """

    model.train()

    total_loss = 0.0
    for i, (X, y) in enumerate(dataloader):
        X = pack(X)
        X = to_device(model.device, X)
        y = to_device(model.device, y)

        total_loss += logger.step(trainer.step(model(*X), y))

    trainer.end_epoch()

    return total_loss / len(dataloader)


@eval_no_grad
def compute_avg_loss(
    model: nn.Module,
    dataloader: DataLoader,
    trainer: Trainer,
    metrics: list[DynamicMetric],
) -> float:
    """
    Returns average loss.
    """

    total_loss = 0.0
    for i, (X, y) in enumerate(dataloader):
        X = pack(X)
        X = to_device(model.device, X)
        y = to_device(model.device, y)

        predicted = model(*X)

        total_loss += trainer.get_loss(predicted, y)

        for metric in metrics:
            metric.update(X=X, predicted=predicted, y=y)

    for metric in metrics:
        metric.reset()

    return total_loss / len(dataloader) if len(dataloader) > 0 else 0.0


def train(
    model: nn.Module,
    dataset: Dataset,
    config: TrainConfig,
    trainer: Trainer,
    metrics: list[DynamicMetric],
    loss_loggers_config: TrainLossLoggerConfig = TrainLossLoggerConfig(),
) -> None:

    train_loader, valid_loader = random_split_dataset(dataset, config, to_loaders=True)
    model.to(config.device)

    clear_dir(loss_loggers_config.log_dir)
    train_epoch_logger = Logger(
        LoggerConfig(
            tb_logs=loss_loggers_config.tb_logs,
            log_dir=loss_loggers_config.log_dir,
            group="Loss/epoch/train",
            text_logs=loss_loggers_config.text_logs,
            title=("train, epoch", "loss"),
            tqdm_total=config.epochs,
            tqdm_position=0,
        )
    )

    valid_epoch_logger = None
    if config.valid_share is not None:
        valid_epoch_logger = Logger(
            LoggerConfig(
                tb_logs=loss_loggers_config.tb_logs,
                log_dir=loss_loggers_config.log_dir,
                group="Loss/epoch/valid",
                text_logs=loss_loggers_config.text_logs,
                title=("valid, epoch", "loss"),
                tqdm_total=config.epochs,
                tqdm_position=1,
            )
        )
    train_step_logger = Logger(
        LoggerConfig(
            log_interval=loss_loggers_config.log_interval,
            reduction="mean",
            tb_logs=loss_loggers_config.tb_logs & loss_loggers_config.step_logs,
            log_dir=loss_loggers_config.log_dir,
            group="Loss/batch/train",
            text_logs=loss_loggers_config.text_logs & loss_loggers_config.step_logs,
            title=("step", "loss"),
            tqdm_position=2,
            tqdm_total=min(
                len(train_loader) * config.epochs,  # type: ignore
                1e9 if config.steps is None else config.steps,
            ),  # type: ignore
        )
    )

    for _ in range(1, config.epochs + 1):
        try:
            train_epoch_logger.step(
                train_one_epoch(
                    model=model,
                    dataloader=train_loader,  # type: ignore
                    trainer=trainer,
                    logger=train_step_logger,
                )
            )
        except StopIteration:
            break

        if config.valid_share is not None:
            assert valid_epoch_logger is not None
            valid_epoch_logger.step(
                compute_avg_loss(
                    model=model,
                    dataloader=valid_loader,  # type: ignore
                    trainer=trainer,
                    metrics=metrics,
                )
            )
