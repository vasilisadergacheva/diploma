import torch
import numpy as np
from torch import Tensor
from typing import Any
from src.metrics import Metrics, RegressionMetrics, ClassificationMetrics


class BaseModel:
    """Abstract class for different approaches."""

    def train(self, X: Any, y: Any, *args, **kwargs) -> "BaseModel":
        raise NotImplementedError

    def predict(self, X: Any, *args, **kwargs) -> Any:
        raise NotImplementedError

    def predict_proba(self, X: Any, *args, **kwargs) -> Any:
        raise NotImplementedError

    def score(self, X: Any, y: Any, *args, **kwargs) -> Metrics:
        raise NotImplementedError

    def clone(self) -> "BaseModel":
        raise NotImplementedError

    def _metric_type(self, y: Any) -> Any:
        if isinstance(y, np.ndarray):
            match y.dtype:
                case np.float32:
                    return RegressionMetrics
                case np.int32:
                    return ClassificationMetrics
                case _:
                    raise ValueError("Can't determine type")
        elif isinstance(y, Tensor):
            match y.dtype:
                case torch.float32:
                    return RegressionMetrics
                case torch.int32:
                    return ClassificationMetrics
                case _:
                    raise ValueError("Can't determine type")
        else:
            print(y)
            raise ValueError("Can't determine type")
