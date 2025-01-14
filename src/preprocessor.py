import torch
import numpy as np
import pandas as pd
from enum import Enum
from typing import Any, Union, Optional
from datetime import timedelta
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.over_sampling import SMOTE

from src.viewer import Viewer
from src.utils import ADMISSION_FEATURES


class DataType(Enum):
    NUMPY = 1
    TORCH = 2
    PANDAS = 3


class TargetType(Enum):
    DEATH = 1
    STAY = 2
    INTENSIVE_STAY = 3

    @staticmethod
    def get_format(target: "TargetType") -> Any:
        return int if target in [TargetType.DEATH] else float


class Preprocessor:
    """Preprocessing the data."""

    def __init__(self, data: pd.DataFrame, target_type: TargetType = TargetType.DEATH):
        self._data = data
        self._target_type = target_type

    @property
    def features_to_create_target(self) -> list[str]:
        return [
            "OUTCOME",
            "MRD No.",
        ] + ADMISSION_FEATURES["time"]

    def _add_target(self, data: pd.DataFrame, trial_period: int) -> pd.DataFrame:
        match self._target_type:
            case TargetType.DEATH:
                data["target"] = (data["OUTCOME"] == "EXPIRY").astype(int)
                data.loc[
                    (data["DATE OF BROUGHT DEAD"] + timedelta(days=trial_period))
                    < data["D.O.D"],
                    "target",
                ] = 1
            case TargetType.STAY:
                data["target"] = data["DURATION OF STAY"]
            case TargetType.INTENSIVE_STAY:
                data["target"] = data["duration of intensive unit stay"]
        return data

    def _convert_type(
        self, data: pd.DataFrame, max_categories: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert types for all columns."""

        # convert numerical features
        _viewer = Viewer(data, max_categories=max_categories)
        data[_viewer.numeric_features] = data[_viewer.numeric_features].astype(float)

        # convert target
        data["target"] = data["target"].astype(TargetType.get_format(self._target_type))

        # factorize categorical features
        for key in _viewer.categorical_features:
            if key == "target":
                continue

            data[key] = pd.factorize(data[key])[0].astype(int)

        return (data[list(set(data.columns) - set(["target"]))], data[["target"]])

    @staticmethod
    def _split_types(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        _viewer = Viewer(data)
        return data[_viewer.categorical_features], data[_viewer.numeric_features]

    def preprocess(
        self,
        trial_period: int = 30,
        to_drop: list[str] = ["EF", "BNP"],
        defaults: dict[str, Any] = {},
        max_categories: int = 7,
        data_type: DataType = DataType.PANDAS,
        balance: Optional[BaseOverSampler] = SMOTE(random_state=42),
        concatenate: bool = True,
    ) -> tuple[Union[Any, tuple[Any, Any]], Any]:
        """Preprocess data"""

        # removing what is required
        data = self._data.drop(
            columns=list(set(to_drop) - set(self.features_to_create_target))
        )

        # add target
        self._add_target(data=data, trial_period=trial_period)
        data.drop(columns=self.features_to_create_target, inplace=True)

        # replacing nans with default values
        for key, value in defaults:
            data[key] = data[key].fillna(value=value)

        # drop nans
        data.dropna(inplace=True)

        # convert types
        features, target = self._convert_type(data=data, max_categories=max_categories)

        # stratify dataset
        if balance and (TargetType.get_format(self._target_type) is int):
            features, target = balance.fit_resample(X=features, y=target)  # type: ignore

        # split or concatenate categorical and numeric features
        categorical, numeric = Preprocessor._split_types(data=features)  # type: ignore
        features = features if concatenate else (categorical, numeric)

        if data_type == DataType.PANDAS:
            return features, target

        numpy_features = (
            np.array(features, dtype=np.float32)
            if concatenate
            else (
                np.array(categorical, dtype=np.int32),
                np.array(numeric, dtype=np.float32),
            )
        )
        numpy_target = np.array(
            target,
            dtype=(
                np.int32
                if TargetType.get_format(self._target_type) is int
                else np.float32
            ),
        ).squeeze()

        match data_type:
            case DataType.NUMPY:
                return numpy_features, numpy_target
            case DataType.TORCH:
                return (
                    torch.from_numpy(numpy_features)
                    if concatenate
                    else (torch.from_numpy(categorical), torch.from_numpy(numeric))
                ), torch.from_numpy(numpy_target)
