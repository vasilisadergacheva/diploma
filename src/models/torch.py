from torch.nn import Module
from torch import Tensor

from src.models.base import BaseModel
from src.models.nn.train import train as _train
from src.models.nn.train import TrainConfig
from src.models.nn.trainers import Trainer


class TorchModel(BaseModel):
    def __init__(self, model: Module) -> None:
        self._model = model

    def train(
        self, X: tuple[Tensor, Tensor], y: Tensor, config: TrainConfig, trainer: Trainer
    ) -> "TorchModel":
        raise NotImplementedError
