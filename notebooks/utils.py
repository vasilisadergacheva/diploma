import os
import matplotlib.pyplot as plt
from pandas import DataFrame
from typing import Optional, Union, Any, Callable
from sklearn.model_selection import train_test_split
from matplotlib.figure import Figure

from src.loader import Loader
from src.preprocessor import Preprocessor, TargetType
from src.metrics import Metrics
from src.models.base import BaseModel


class DataPipeline:
    """Basic data loading pipeline"""

    # load_raw
    loader: Optional[Loader] = None

    raw_data: Optional[DataFrame] = None
    raw_preprocessor: Optional[Preprocessor] = None

    # train_test_split
    train_data: Optional[DataFrame] = None
    test_data: Optional[DataFrame] = None

    # get_preprocessors
    targets: Optional[list[TargetType]] = None
    train_preprocessors: Optional[dict[TargetType, Preprocessor]] = None
    test_preprocessors: Optional[dict[TargetType, Preprocessor]] = None

    def load_raw(self) -> None:
        self.loader = Loader("data")
        self.raw_data = self.loader.load()

        self.raw_preprocessor = Preprocessor(self.raw_data)

    def train_test_split(self) -> None:
        if self.raw_data is None or self.raw_preprocessor is None:
            raise ValueError("load raw first")

        raw_data_for_stratify = self.raw_preprocessor._add_target(
            data=self.raw_data.copy(), trial_period=30
        )

        train_data, test_data = train_test_split(
            self.raw_data,
            test_size=0.2,
            stratify=raw_data_for_stratify["target"],
            shuffle=True,
        )

        self.train_data = train_data
        self.test_data = test_data

    def get_preprocessors(self) -> None:
        if self.train_data is None or self.test_data is None:
            raise ValueError("split train/test first")

        self.targets = [target for target in TargetType]

        self.train_preprocessors = dict(
            (
                target,
                Preprocessor(self.train_data, target_type=target),
            )
            for target in self.targets
        )

        self.test_preprocessors = dict(
            (
                target,
                Preprocessor(self.test_data, target_type=target),
            )
            for target in self.targets
        )

    def __call__(self) -> None:
        self.load_raw()
        self.train_test_split()
        self.get_preprocessors()

    def preprocess(self, *args, **kwargs) -> tuple[
        dict[TargetType, tuple[Union[Any, tuple[Any, Any]], Any]],
        dict[TargetType, tuple[Union[Any, tuple[Any, Any]], Any]],
    ]:
        _preprocess_func = lambda prerocessor: prerocessor.preprocess(*args, **kwargs)
        _apply_to_dict = lambda _dict: dict(
            zip(_dict, map(_preprocess_func, _dict.values()))
        )

        return _apply_to_dict(self.train_preprocessors), _apply_to_dict(
            self.test_preprocessors
        )


class ModelsEnsemble:
    def __init__(self, models: dict[TargetType, list[BaseModel]] = {}):
        self.models = models

    def __len__(self):
        return len(list(self.models.values())[0])

    @classmethod
    def single_target_from_parameters(
        cls: Any,
        target: TargetType,
        model_class: Any,
        constructor_class: Any,
        parameters: dict[str, list[Any]],
    ) -> "ModelsEnsemble":
        _number_models = len(list(parameters.values())[0])
        parameters_as_dict = [dict() for _ in range(_number_models)]

        for parameter, values in parameters.items():
            if len(values) != _number_models:
                raise ValueError("provide all parameters for each model")

            for i in range(_number_models):
                parameters_as_dict[i][parameter] = values[i]

        return cls(
            models={
                target: [
                    model_class(constructor_class(**params))
                    for params in parameters_as_dict
                ]
            }
        )

    @classmethod
    def from_parameters(
        cls: Any,
        model_class: Any,
        constructor_class: Union[Any, dict[TargetType, Any]],
        parameters: dict[TargetType, dict[str, list[Any]]],
    ) -> "ModelsEnsemble":
        return cls(
            models=dict(
                (
                    target,
                    cls.single_target_from_parameters(
                        target=target,
                        model_class=model_class,
                        constructor_class=(
                            constructor_class[target]
                            if isinstance(constructor_class, dict)
                            else constructor_class
                        ),
                        parameters=target_params,
                    ).models[target],
                )
                for target, target_params in parameters.items()
            )
        )

    def train(self, data: dict[TargetType, Any]) -> None:
        for target in self.models.keys():
            for model in self.models[target]:
                model.train(*data[target])

    def metrics(self, data: dict[TargetType, Any]) -> dict[TargetType, list[Metrics]]:
        return dict(
            (target, [model.score(*data[target]) for model in self.models[target]])
            for target in self.models.keys()
        )


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
        test_data: dict[TargetType, Any],
        train_data: Optional[dict[TargetType, Any]] = None,
        x: Optional[list[Any]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        name_suffix: str = "",
        add_train_metrics: bool = True,
    ) -> None:
        if not (train_data is None):
            ensemble.train(data=train_data)

        def _get_metrics(data: dict[TargetType, Any]) -> dict[TargetType, list[Any]]:
            metrics: dict[TargetType, list[Metrics]] = ensemble.metrics(data=data)
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

            plt.xticks(x)
            plt.legend()

            if not (xlabel is None):
                plt.xlabel(xlabel=xlabel)

            if not (ylabel is None):
                plt.ylabel(ylabel=ylabel)

            name = "_".join([self.name, target.name])
            plt.title(name + name_suffix)
            plt.savefig(os.path.join(self.dirname, name))
