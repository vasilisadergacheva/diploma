import numpy as np
from typing import Any, Union
from sklearn.metrics import accuracy_score, r2_score, precision_score, recall_score
from pydantic import BaseModel, Field, ConfigDict


class ClassificationMetrics(BaseModel):
    accuracy: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    model_config = ConfigDict(frozen=True)

    @staticmethod
    def from_ndarrays(predicted: np.ndarray, y: np.ndarray) -> "ClassificationMetrics":
        _sklearn_score = lambda scoring_function: float(
            scoring_function(y_true=y, y_pred=predicted)
        )

        return ClassificationMetrics(
            accuracy=_sklearn_score(accuracy_score),
            precision=_sklearn_score(precision_score),
            recall=_sklearn_score(recall_score),
        )


class RegressionMetrics(BaseModel):
    r2: float = Field(le=1)

    @staticmethod
    def from_ndarrays(predicted: np.ndarray, y: np.ndarray) -> "RegressionMetrics":
        _sklearn_score = lambda scoring_function: float(
            scoring_function(y_true=y, y_pred=predicted)
        )

        return RegressionMetrics(
            r2=_sklearn_score(r2_score),
        )


Metrics = Union[ClassificationMetrics, RegressionMetrics]


class DynamicMetric:
    def update(self, X: Any, predicted: Any, y: Any) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
