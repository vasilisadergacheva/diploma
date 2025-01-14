import numpy as np
from typing import Any

from src.models.base import BaseModel
from src.metrics import Metrics


class SklearnModel(BaseModel):
    """Sklearn models."""

    def __init__(self, estimator: Any):
        self._estimator = estimator

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._estimator.fit(X=X, y=y)
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._estimator.predict(X=X)

    def score(self, X: np.ndarray, y: np.ndarray) -> Metrics:
        return self._metric_type(y).from_ndarrays(predicted=self.predict(X), y=y)
