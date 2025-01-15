import numpy as np
from typing import Any, Union, Optional
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from pydantic import BaseModel, Field, ConfigDict

from src.utils import fbeta


class ClassificationMetrics(BaseModel):
    accuracy: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    roc_auc_score: Optional[float] = Field(ge=0, le=1, default=None)
    model_config = ConfigDict(frozen=True)

    @staticmethod
    def from_ndarrays(
        predicted: np.ndarray, y: np.ndarray, proba: Optional[np.ndarray] = None
    ) -> "ClassificationMetrics":
        _sklearn_score = lambda scoring_function: float(
            scoring_function(y_true=y, y_pred=predicted)
        )

        _sklearn_score_with_proba = lambda scoring_function: (
            scoring_function(y_true=y, y_score=proba[:, 1])
            if not (proba is None)
            else None
        )

        return ClassificationMetrics(
            accuracy=_sklearn_score(accuracy_score),
            precision=_sklearn_score(precision_score),
            recall=_sklearn_score(recall_score),
            f1=_sklearn_score(f1_score),
            roc_auc_score=_sklearn_score_with_proba(roc_auc_score),
        )

    @staticmethod
    def aggregate(metrics: list["ClassificationMetrics"]) -> "ClassificationMetrics":
        _aggregate = lambda extractor, aggregator: aggregator(
            [extractor(x) for x in metrics]
        )

        _mean_precision = _aggregate(lambda x: x.precision, np.mean)
        _mean_recall = _aggregate(lambda x: x.recall, np.mean)

        return ClassificationMetrics(
            accuracy=_aggregate(lambda x: x.accuracy, np.mean),
            precision=_mean_precision,
            recall=_mean_recall,
            f1=fbeta(precision=_mean_precision, recall=_mean_recall),
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

    @staticmethod
    def aggregate(metrics: list["RegressionMetrics"]) -> "RegressionMetrics":
        raise NotImplementedError


Metrics = Union[ClassificationMetrics, RegressionMetrics]


def aggregate(metrics: list[Metrics]) -> Metrics:
    return (
        ClassificationMetrics.aggregate(metrics=metrics)  # type: ignore
        if isinstance(metrics[0], ClassificationMetrics)
        else RegressionMetrics.aggregate(metrics=metrics)  # type: ignore
    )


class DynamicMetric:
    def update(self, X: Any, predicted: Any, y: Any) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
