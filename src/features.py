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

    def _compute_target_direction(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        column = kwargs.get("column", "close")
        df["target_direction"] = (df[column].shift(-1) > df[column]).astype(int)

        return df

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

    def fit(self, data: pd.DataFrame, target_col: str):
        self.scaler = StandardScaler()
        self.features_to_scale = [col for col in data.columns if col != target_col]
        self.scaler.fit(data[self.features_to_scale])

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result[self.features_to_scale] = self.scaler.transform(data[self.features_to_scale])
        return result

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result[self.features_to_scale] = self.scaler.inverse_transform(data[self.features_to_scale])
        return result

    def fit_transorm(self, data: pd.DataFrame, target_col: str):
        self.fit(data, target_col)
        data_scaled = self.transform(data)

        return data_scaled
