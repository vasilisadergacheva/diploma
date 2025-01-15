from typing import Union, Any
from sklearn.base import clone
from joblib import delayed, Parallel

from src.preprocessor import TargetType
from src.metrics import Metrics, aggregate
from src.models.base import BaseModel
from notebooks.src.utils import apply


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

    def train(self, data: dict[TargetType, Any], n_jobs: int = 4) -> "ModelsEnsemble":
        for target in self.models.keys():
            _train_model = lambda model: model.train(*data[target])
            self.models[target] = list(  # type: ignore
                Parallel(n_jobs=n_jobs)(
                    [delayed(_train_model)(model) for model in self.models[target]]
                )
            )
        return self

    def metrics(self, data: dict[TargetType, Any]) -> dict[TargetType, list[Metrics]]:
        return dict(
            (target, [model.score(*data[target]) for model in self.models[target]])
            for target in self.models.keys()
        )

    def clone(self) -> "ModelsEnsemble":
        return ModelsEnsemble(
            models=apply(
                self.models,
                lambda target_models: [model.clone() for model in target_models],
            )
        )

    @staticmethod
    def train_many(
        ensemble: "ModelsEnsemble",
        datasets: list[dict[TargetType, Any]],
        n_jobs: int = 4,
    ) -> list["ModelsEnsemble"]:
        return [ensemble.clone().train(data=data, n_jobs=n_jobs) for data in datasets]

    @staticmethod
    def metrics_many(
        ensembles: list["ModelsEnsemble"], datasets: list[dict[TargetType, Any]]
    ) -> dict[TargetType, list[Metrics]]:
        metrics: list[dict[TargetType, list[Metrics]]] = [
            ensemble.metrics(data) for ensemble, data in zip(ensembles, datasets)
        ]
        return dict(
            (
                target,
                [
                    aggregate([metric[target][i] for metric in metrics])
                    for i in range(len(metrics[0][target]))
                ],
            )
            for target in metrics[0].keys()
        )
