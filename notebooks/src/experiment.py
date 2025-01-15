import os
import matplotlib.pyplot as plt
from typing import Optional, Any, Callable
from matplotlib.figure import Figure

from src.preprocessor import TargetType
from src.metrics import Metrics
from notebooks.src.ensemble import ModelsEnsemble


class Experiment:
    def __init__(
        self, name: str, figures: list[Figure] = [], dirname: str = "notebooks/plots"
    ) -> None:
        self.name = name
        self.figures = figures
        self.dirname = dirname

    def model_ensemble(
        self,
        ensemble: ModelsEnsemble,
        metric_map: Callable,
        test_data: list[dict[TargetType, Any]],
        train_data: Optional[list[dict[TargetType, Any]]] = None,
        x: Optional[list[Any]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        name_suffix: str = "",
        add_train_metrics: bool = True,
        n_jobs: int = 4,
    ) -> None:
        if not (train_data is None):
            ensembles = ModelsEnsemble.train_many(
                ensemble=ensemble, datasets=train_data, n_jobs=n_jobs
            )
        else:
            ensembles = [ensemble.clone() for _ in range(len(test_data))]

        def _get_metrics(
            data: list[dict[TargetType, Any]]
        ) -> dict[TargetType, list[Any]]:
            metrics: dict[TargetType, list[Metrics]] = ModelsEnsemble.metrics_many(
                ensembles=ensembles, datasets=data
            )
            values: dict[TargetType, list[Any]] = dict(
                zip(metrics, map(lambda x: list(map(metric_map, x)), metrics.values()))
            )
            return values

        test_metrics = _get_metrics(data=test_data)
        if add_train_metrics:
            if train_data is None:
                raise ValueError("provide train_data")

            train_metrics = _get_metrics(data=train_data)

        if x is None:
            x = list(range(len(ensemble)))

        for target in ensemble.models.keys():
            plt.plot(x, test_metrics[target], label="test metric")

            if add_train_metrics:
                plt.plot(x, train_metrics[target], label="train metric")  # type: ignore

            if len(x) <= 5:  # TODO: magic number
                plt.xticks(x)

            plt.legend()

            if not (xlabel is None):
                plt.xlabel(xlabel=xlabel)

            if not (ylabel is None):
                plt.ylabel(ylabel=ylabel)

            name = "_".join([self.name, target.name])
            plt.title(name + name_suffix)
            plt.savefig(os.path.join(self.dirname, name))
            plt.show()
