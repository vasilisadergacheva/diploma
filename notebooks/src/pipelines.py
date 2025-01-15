import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union, Any
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.impute._base import _BaseImputer
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler


from src.viewer import Viewer
from src.loader import Loader
from src.preprocessor import Preprocessor, TargetType, DataType
from src.utils import _extract_columns
from notebooks.src.utils import apply


class DataPipeline:
    """Basic data loading pipeline"""

    # load_raw
    loader: Optional[Loader] = None

    raw_data: Optional[DataFrame] = None

    # splits
    train_data: Optional[list[DataFrame]] = None
    test_data: Optional[list[DataFrame]] = None

    # get_preprocessors
    targets: Optional[list[TargetType]] = None
    train_preprocessors: Optional[list[dict[TargetType, Preprocessor]]] = None
    test_preprocessors: Optional[list[dict[TargetType, Preprocessor]]] = None

    def load_raw(
        self,
        random_seed: int = 42,
        add_pollution: bool = False,
        to_drop: list[str] = [],
        dropna: bool = True,
    ) -> None:
        np.random.seed(random_seed)

        self.loader = Loader("data")
        self.raw_data = Preprocessor(
            self.loader.load(add_pollution=add_pollution)
        ).drop(to_drop=to_drop, dropna=dropna)

    def train_test_split(self, test_size: float = 0.2) -> None:
        if self.raw_data is None:
            raise ValueError("load raw first")

        # copy data
        data = self.raw_data.copy()

        train_data, test_data = train_test_split(
            data,
            test_size=test_size,
            stratify=data["target"],
            shuffle=True,
        )

        self.train_data = [train_data]
        self.test_data = [test_data]

    def kfold_split(self, n_splits: int = 2):
        if self.raw_data is None:
            raise ValueError("load raw first")

        # copy data
        data = self.raw_data.copy()

        skf = StratifiedKFold(n_splits=n_splits)

        features, target = _extract_columns(data=data, save=True)

        self.train_data = []
        self.test_data = []
        for i, (train_index, test_index) in enumerate(skf.split(X=features, y=target)):
            self.train_data.append(features.iloc[train_index].copy())
            self.test_data.append(features.iloc[test_index].copy())

    def get_preprocessors(self) -> None:
        if self.train_data is None or self.test_data is None:
            raise ValueError("split train/test first")

        self.targets = [target for target in TargetType]

        _to_preprocessors = lambda data: [
            dict(
                (
                    target,
                    Preprocessor(_data, target_type=target),
                )
                for target in self.targets  # type: ignore
            )
            for _data in data
        ]

        self.train_preprocessors = _to_preprocessors(self.train_data)
        self.test_preprocessors = _to_preprocessors(self.test_data)

    def __call__(
        self,
        add_pollution: bool = True,
        random_seed: int = 42,
        test_size: float = 0.2,
        n_splits: int = 1,
        to_drop: list[str] = [],
        dropna: bool = False,
    ) -> None:
        self.load_raw(
            add_pollution=add_pollution,
            random_seed=random_seed,
            to_drop=to_drop,
            dropna=dropna,
        )
        if n_splits == 1:
            self.train_test_split(test_size=test_size)
        else:
            self.kfold_split(n_splits=n_splits)
        if not dropna:
            self.impute()
        self.get_preprocessors()

    def preprocess(
        self,
        max_categories: int = 7,
        data_type: DataType = DataType.PANDAS,
        train_balance: Optional[BaseOverSampler] = RandomOverSampler(),
        concatenate: bool = True,
    ) -> tuple[
        list[dict[TargetType, tuple[Union[Any, tuple[Any, Any]], Any]]],
        list[dict[TargetType, tuple[Union[Any, tuple[Any, Any]], Any]]],
    ]:
        def _preprocess(
            data: list[dict[TargetType, Preprocessor]], _balance
        ) -> list[dict[TargetType, tuple[Union[Any, tuple[Any, Any]], Any]]]:
            return [
                apply(
                    _dict,
                    lambda preprocessor: preprocessor.preprocess(
                        max_categories=max_categories,
                        data_type=data_type,
                        balance=_balance,
                        concatenate=concatenate,
                    ),
                )
                for _dict in data
            ]

        return _preprocess(
            data=self.train_preprocessors, _balance=train_balance  # type: ignore
        ), _preprocess(
            data=self.test_preprocessors, _balance=None  # type: ignore
        )

    def impute(
        self,
        numeric_imputer: Optional[_BaseImputer] = None,
        categorical_imputer: Optional[_BaseImputer] = None,
        scaler: Optional[Any] = None,
        onehot_encoder: Optional[Any] = None,
    ):
        if self.train_data is None or self.test_data is None:
            raise ValueError

        for i in range(len(self.train_data)):
            if numeric_imputer is None:
                numeric_imputer = IterativeImputer(initial_strategy="median")

            if categorical_imputer is None:
                categorical_imputer = SimpleImputer(strategy="most_frequent")

            if scaler is None:
                scaler = StandardScaler()

            if onehot_encoder is None:
                onehot_encoder = (
                    OneHotEncoder(
                        handle_unknown="ignore", drop="if_binary", sparse_output=False
                    ),
                )

            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", numeric_imputer),
                    ("scaler", scaler),
                ]
            )
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", categorical_imputer),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            drop="if_binary",
                            sparse_output=False,
                        ),
                    ),
                ]
            )

            X_train, y_train = _extract_columns(
                data=self.train_data[i], columns=["target"] + Preprocessor.targets()
            )
            X_test, y_test = _extract_columns(
                data=self.test_data[i], columns=["target"] + Preprocessor.targets()
            )

            _viewer = Viewer(X_train)

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, _viewer.numeric_features),
                    (
                        "cat",
                        categorical_transformer,
                        _viewer.categorical_features,
                    ),
                ],
                remainder="passthrough",
            )

            X_train = pd.DataFrame(
                preprocessor.fit_transform(X_train),  # type: ignore
                columns=[
                    x.replace("num__", "").replace("cat__", "")
                    for x in preprocessor.get_feature_names_out()
                ],
                index=X_train.index,
            )
            X_test = pd.DataFrame(
                preprocessor.transform(X_test),  # type: ignore
                columns=[
                    x.replace("num__", "").replace("cat__", "")
                    for x in preprocessor.get_feature_names_out()
                ],
                index=X_test.index,
            )

            self.train_data[i] = X_train.merge(  # type: ignore
                y_train, left_index=True, right_index=True
            )

            self.test_data[i] = X_test.merge(y_test, left_index=True, right_index=True)  # type: ignore
