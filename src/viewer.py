import pandas as pd
import seaborn as sns


class Viewer:
    """A data analysis tool."""

    def __init__(self, data: pd.DataFrame, max_categories: int = 7):
        self._data = data
        self.max_categories = max_categories

    @property
    def features(self) -> list[str]:
        return list(self._data.columns)

    @property
    def datetime_features(self) -> list[str]:
        return ["D.O.A", "D.O.D", "DATE OF BROUGHT DEAD"]

    @property
    def categorical_features(self) -> list[str]:
        return [
            key
            for key in self._data.columns
            if self._is_feature_categorical(key) and (key not in self.datetime_features)
        ]

    @property
    def numeric_features(self) -> list[str]:
        return [
            key
            for key in self._data.columns
            if (not self._is_feature_categorical(key))
            and (key not in self.datetime_features)
        ]

    def check_nans(self) -> None:
        """Make sure that numeric features can be converted to float. \
            datetime features are datetime, \
            And check the unique values of the categorical features."""

        # datetime features
        for key in self.datetime_features:
            assert self._data[key].dtypes == "datetime64[ns]", f"{key} is not datetime"

        # numeric features
        for key in self.numeric_features:
            try:
                with pd.option_context(
                    "future.no_silent_downcasting", True
                ):  # no warning
                    self._data[key].fillna(0).astype(float)
            except Exception as exception:
                print(f"feature: {key}, exception: \n", exception)

        # categorical features
        for key in self.categorical_features:
            values = list(self._data[key].unique())
            print(f"feature: {key}, \n unique_values: {values}\n")

    def describe_feature(self, key: str) -> None:
        """Plot information about feature."""
        nans_count = sum(self._data[key].isna())
        print(f"nans: {nans_count}")
        if self._is_feature_categorical(key):
            sns.displot(data=self._data, x=key).set_titles(key)
        else:
            sns.kdeplot(data=list(self._data[key].astype(float))).set_title(key)

    def nans_count(self) -> dict[str, int]:
        """Get nans count for each feature."""
        nans_count = [(key, sum(self._data[key].isna())) for key in self.features]
        return dict([x for x in nans_count if x[1] > 0])

    def _is_feature_categorical(self, key: str) -> bool:
        """Check if feature is categorical"""
        return len(self._data[key].unique()) <= self.max_categories
