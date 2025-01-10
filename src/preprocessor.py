import torch
import numpy as np
import pandas as pd
from enum import Enum
from typing import Any, Union
from datetime import timedelta

from src.viewer import Viewer
from src.utils import ADMISSION_FEATURES


class DataType(Enum):
    NUMPY = 1
    TORCH = 2
    PANDAS = 3


class Preprocessor:
    """Preprocessing the data."""

    def __init__(
        self,
        data: pd.DataFrame,
    ):
        self._data = data

    @property
    def features_to_create_target(self) -> list[str]:
        return [
            "OUTCOME",
            "MRD No.",
        ] + ADMISSION_FEATURES["time"]

    @staticmethod
    def _add_target(data: pd.DataFrame, trial_period: int) -> pd.DataFrame:
        data["target"] = (data["OUTCOME"] == "EXPIRY").astype(int)
        data.loc[
            (data["DATE OF BROUGHT DEAD"] + timedelta(days=trial_period))
            < data["D.O.D"],
            "target",
        ] = 1
        return data

    @staticmethod
    def _convert_type(
        data: pd.DataFrame, max_categories: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convert types for all columns."""

        # convert numerical features
        _viewer = Viewer(data, max_categories=max_categories)
        data[_viewer.numeric_features] = data[_viewer.numeric_features].astype(float)

        # factorize categorical features
        for key in _viewer.categorical_features:
            data[key] = (
                data[key].astype(int)
                if key == "target"
                else pd.factorize(data[key])[0].astype(int)
            )

        return (
            data[list(set(_viewer.categorical_features) - set(["target"]))],
            data[_viewer.numeric_features],
            data[["target"]],
        )

    def preprocess(
        self,
        trial_period: int = 30,
        to_drop: list[str] = [],
        defaults: dict[str, Any] = {},
        max_categories: int = 7,
        data_type: DataType = DataType.PANDAS,
        concatenate: bool = True,
    ) -> tuple[Union[Any, tuple[Any, Any]], Any]:
        """Preprocess data"""

        # removing what is required
        data = self._data.drop(
            columns=list(set(to_drop) - set(self.features_to_create_target))
        )

        # add target
        Preprocessor._add_target(data=data, trial_period=trial_period)
        data.drop(columns=self.features_to_create_target, inplace=True)

        # replacing nans with default values
        for key, value in defaults:
            data[key] = data[key].fillna(value=value)

        # drop nans
        data.dropna(inplace=True)

        # convert types
        categorical, numeric, target = Preprocessor._convert_type(
            data=data, max_categories=max_categories
        )

        # concatenate categorical and numeric features if needed
        features = (
            pd.merge(left=categorical, right=numeric, left_index=True, right_index=True)
            if concatenate
            else (categorical, numeric)
        )

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
        numpy_target = np.array(target, dtype=np.int32)

        match data_type:
            case DataType.NUMPY:
                return numpy_features, target
            case DataType.TORCH:
                return (
                    torch.from_numpy(numpy_features)
                    if concatenate
                    else (torch.from_numpy(categorical), torch.from_numpy(numeric))
                ), torch.from_numpy(numpy_target)
