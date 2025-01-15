import numpy as np
from sklearn.base import clone
from typing import Any

from src.models.base import BaseModel
from src.metrics import Metrics


class SklearnModel(BaseModel):
    """Sklearn models."""

    def __init__(self, estimator: Any):
        self._estimator = estimator

    def train(self, X: np.ndarray, y: np.ndarray) -> "SklearnModel":
        self._estimator.fit(X=X, y=y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._estimator.predict(X=X)

    def predict_proba(self, X: Any, *args, **kwargs) -> Any:
        return self._estimator.predict_proba(X=X, *args, **kwargs)

    def score(self, X: np.ndarray, y: np.ndarray) -> Metrics:
        try:
            proba = self.predict_proba(X=X)
        except Exception:
            proba = None
        return self._metric_type(y).from_ndarrays(
            predicted=self.predict(X), y=y, proba=proba
        )

    def clone(self) -> "SklearnModel":
        return SklearnModel(estimator=clone(self._estimator))
