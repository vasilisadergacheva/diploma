import os
import pandas as pd
import numpy as np
from typing import Any
from datetime import datetime
from itertools import product


class Loader:
    """Loads and preprocess data."""

    def __init__(self, path: str) -> None:
        self.path: str = path

    @staticmethod
    def _fix_date(admission: pd.DataFrame) -> pd.DataFrame:
        """fix admission dates
        possible causes of the error: arithmetic, in case of multiple visit or something unknown
        """

        def fix_date(row: pd.Series) -> pd.Series:
            date_formats = ["%d/%m/%Y", "%m/%d/%Y"]
            possible_dates = set()

            for format in product(date_formats, date_formats):
                admission_date_format, discharge_date_format = format

                try:
                    admission_date = datetime.strptime(
                        row["D.O.A"], admission_date_format
                    )
                except:
                    continue

                try:
                    discharge_date = datetime.strptime(
                        row["D.O.D"], discharge_date_format
                    )
                except:
                    continue

                admission_month = datetime.strptime(row["month year"], "%b-%y")

                if (
                    admission_date.month != admission_month.month
                    or admission_date.year != admission_month.year
                ):
                    continue

                gap = (discharge_date - admission_date).days + 1

                if gap < 0 or gap != row["DURATION OF STAY"]:
                    continue

                possible_dates.add((admission_date, discharge_date))

            if len(possible_dates) != 1:
                row["D.O.A"] = np.nan
                row["D.O.D"] = np.nan
            else:
                admission_date, discharge_date = possible_dates.pop()
                row["D.O.A"] = admission_date
                row["D.O.D"] = discharge_date
            return row

        admission = admission.transform(fix_date, axis=1)
        admission["D.O.A"] = admission["D.O.A"].astype("datetime64[ns]")
        admission["D.O.D"] = admission["D.O.A"].astype("datetime64[ns]")
        admission = admission[admission["D.O.A"].notna() & admission["D.O.D"].notna()]

        return admission

    def _replace_nans(self, admission: pd.DataFrame, nans: list[Any]) -> pd.DataFrame:
        with pd.option_context("future.no_silent_downcasting", True):  # no warning
            admission.replace(nans, value=np.nan, inplace=True)

        return admission

    def load(
        self,
        add_mortality: bool = True,
        add_pollution: bool = False,
        nans: list[Any] = ["EMPTY", "\\", "NILL", "`150", "6+2", "S"],
    ) -> pd.DataFrame:
        """Loads data."""
        # read data
        admission: pd.DataFrame = pd.read_csv(
            os.path.join(self.path, "HDHI Admission data.csv")
        )
        mortality: pd.DataFrame = pd.read_csv(
            os.path.join(self.path, "HDHI Mortality Data.csv")
        )
        pollution: pd.DataFrame = pd.read_csv(
            os.path.join(self.path, "HDHI Pollution Data.csv")
        )

        admission = Loader._fix_date(admission)

        # if necessary, add information about mortality
        if add_mortality:
            # convert to datetime
            mortality["DATE OF BROUGHT DEAD"] = mortality["DATE OF BROUGHT DEAD"].apply(
                lambda x: datetime.strptime(x, "%m/%d/%Y")
            )

            # join on MRD
            admission = admission.merge(
                right=mortality[["MRD", "DATE OF BROUGHT DEAD"]],
                how="left",
                left_on="MRD No.",
                right_on="MRD",
            ).drop(columns=["MRD"])

        # if necessary, add information about pollution
        if add_pollution:
            pollution["DATE"] = pollution["DATE"].apply(
                lambda x: datetime.strptime(x, "%m/%d/%Y")
            )
            admission = admission.merge(
                right=pollution, how="left", left_on="D.O.A", right_on="DATE"
            ).drop(columns=["DATE"])

        # select only the last appointment for each patient
        admission = admission.loc[
            admission.groupby("MRD No.")["D.O.A"].idxmax()
        ].reset_index()

        # drop unnecessary columns
        admission.drop(columns=["index", "SNO", "month year"], inplace=True)

        # EMPTY and other literals to NaN
        admission = self._replace_nans(admission=admission, nans=nans)

        return admission
