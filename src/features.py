from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, features: dict[str, dict[str, Any]]):
        self.features = features
        self.scaler = None

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        for key, value in self.features.items():
            method_name = f"_compute_{key}"

            if hasattr(self, method_name):
                func = getattr(self, method_name)
                data = func(data, **value)
            else:
                raise ValueError(f"No function for key: {key}")

        return data

    def _compute_log_return(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        column = kwargs.get("column", "close")
        periods = kwargs.get("periods", [1])

        if not isinstance(periods, list):
            periods = [periods]

        for period in periods:
            col_name = f"log_return_{period}"
            df[col_name] = np.log(df[column] / df[column].shift(period))

        return df

    def _compute_rsi(self, df: pd.DataFrame, **kwargs):
        column = kwargs.get("column", "close")
        periods = kwargs.get("periods", [14])

        if not isinstance(periods, list):
            periods = [periods]

        delta = df[column].diff()

        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)

        for period in periods:
            ema_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
            ema_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

            rs = ema_gain / ema_loss
            col_name = f"rsi_{period}"

            df[col_name] = 100 - (100 / (1 + rs))

        return df

    def fit(self, data: pd.DataFrame):
        self.scaler = StandardScaler()
        self.scaler.fit(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        scaled_data = self.scaler.transform(data)
        return pd.DataFrame(scaled_data, index=data.index, columns=data.columns)

    def fit_transorm(self, data: pd.DataFrame):
        self.fit(data)
        data_scaled = self.transform(data)

        return data_scaled
