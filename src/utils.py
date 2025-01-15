import pandas as pd

ADMISSION_FEATURES = {
    "polllution": [
        "AQI",
        "PM2.5 AVG",
        "PM2.5 MIN",
        "PM2.5 MAX",
        "PM10 AVG",
        "PM10 MIN",
        "PM10 MAX",
        "NO2 AVG",
        "NO2 MIN",
        "NO2 MAX",
        "NH3 AVG",
        "NH3 MIN",
        "NH3 MAX",
        "SO2 AVG",
        "SO2 MIN",
        "SO2 MAX",
        "CO AVG",
        "CO MIN",
        "CO MAX",
        "OZONE AVG",
        "OZONE MIN",
        "OZONE MAX",
        "PROMINENT POLLUTENT",
        "MAX TEMP",
        "MIN TEMP",
        "HUMIDITY",
    ],
    "medical": [
        "SMOKING ",
        "ALCOHOL",
        "DM",
        "HTN",
        "CAD",
        "PRIOR CMP",
        "CKD",
        "HB",
        "TLC",
        "PLATELETS",
        "GLUCOSE",
        "UREA",
        "CREATININE",
        "BNP",
        "RAISED CARDIAC ENZYMES",
        "EF",
        "SEVERE ANAEMIA",
        "ANAEMIA",
        "STABLE ANGINA",
        "ACS",
        "STEMI",
        "ATYPICAL CHEST PAIN",
        "HEART FAILURE",
        "HFREF",
        "HFNEF",
        "VALVULAR",
        "CHB",
        "SSS",
        "AKI",
        "CVA INFRACT",
        "CVA BLEED",
        "AF",
        "VT",
        "PSVT",
        "CONGENITAL",
        "UTI",
        "NEURO CARDIOGENIC SYNCOPE",
        "ORTHOSTATIC",
        "INFECTIVE ENDOCARDITIS",
        "DVT",
        "CARDIOGENIC SHOCK",
        "SHOCK",
        "PULMONARY EMBOLISM",
        "CHEST INFECTION",
    ],
    "time": [
        "D.O.A",
        "D.O.D",
        "DATE OF BROUGHT DEAD",
        "DURATION OF STAY",
        "duration of intensive unit stay",
    ],
    "demographic": ["AGE", "GENDER", "RURAL"],
    "admission": ["MRD No.", "TYPE OF ADMISSION-EMERGENCY/OPD", "OUTCOME"],
}


def _extract_columns(
    data: pd.DataFrame, columns: list[str] = ["target"], save: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (data if save else data[list(set(data.columns) - set(columns))]), data[
        columns
    ]


def fbeta(precision: float, recall: float, beta: float = 1, eps: float = 1e-5) -> float:
    if precision + recall < eps:
        return 0
    return (1 + (beta**2)) * (precision * recall) / ((beta**2) * precision + recall)
