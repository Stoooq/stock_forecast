from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, features: dict[str, dict[str, Any]]):
        self.features = features
        self.scaler = None

        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.features_cols = None
        self.target_col = None

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

    def _compute_rolling_volatility(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        column = kwargs.get("column", "close")
        periods = kwargs.get("periods", [14])

        if not isinstance(periods, list):
            periods = [periods]

        returns = df[column].pct_change()

        for period in periods:
            col_name = f"volatility_{period}"
            df[col_name] = returns.rolling(window=period).std()

        return df

    def _compute_momentum(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        column = kwargs.get("column", "close")
        periods = kwargs.get("periods", [10])

        if not isinstance(periods, list):
            periods = [periods]

        for period in periods:
            col_name = f"momentum_{period}"
            df[col_name] = df[column].pct_change(periods=period)

        return df

    def _compute_zscore(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        column = kwargs.get("column", "close")
        periods = kwargs.get("periods", [20])

        if not isinstance(periods, list):
            periods = [periods]

        for period in periods:
            col_name = f"zscore_{period}"
            rolling_mean = df[column].rolling(window=period).mean()
            rolling_std = df[column].rolling(window=period).std()
            df[col_name] = (df[column] - rolling_mean) / rolling_std

        return df

    def _compute_macd(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        column = kwargs.get("column", "close")
        fast_period = kwargs.get("fast_period", 12)
        slow_period = kwargs.get("slow_period", 26)
        signal_period = kwargs.get("signal_period", 9)

        ema_fast = df[column].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        col_prefix = f"macd_{fast_period}_{slow_period}"
        df[f"{col_prefix}_line"] = macd_line
        df[f"{col_prefix}_signal"] = signal_line
        df[f"{col_prefix}_hist"] = macd_line - signal_line

        return df

    def _compute_ema(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        column = kwargs.get("column", "close")
        periods = kwargs.get("periods", [10, 20, 50])

        if not isinstance(periods, list):
            periods = [periods]

        for period in periods:
            col_name = f"ema_{period}"
            df[col_name] = df[column].ewm(span=period, adjust=False).mean()

        return df

    def fit(
        self,
        data: pd.DataFrame,
        target_col: str,
        no_scale_cols: list[str] | None = None,
    ):
        if no_scale_cols is None:
            no_scale_cols = []

        self.target_col = target_col
        self.features_to_scale = [
            col
            for col in data.columns
            if col not in no_scale_cols and col != target_col
        ]

        self.feature_scaler.fit(data[self.features_to_scale])

        if target_col not in no_scale_cols:
            self.target_scaler.fit(data[[target_col]])

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()

        if self.features_to_scale:
            result[self.features_to_scale] = self.feature_scaler.transform(
                data[self.features_to_scale],
            )

        if (
            hasattr(self.target_scaler, "scale_")
            and self.target_scaler.scale_ is not None
        ):
            result[self.target_col] = self.target_scaler.transform(
                data[[self.target_col]],
            )

        return result

    def inverse_transform_target(self, target: np.ndarray) -> np.ndarray:
        if (
            not hasattr(self.target_scaler, "scale_")
            or self.target_scaler.scale_ is None
        ):
            return target

        y_pred_reshaped = target.reshape(-1, 1)

        return self.target_scaler.inverse_transform(y_pred_reshaped).flatten()

    def inverse_transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()

        if self.features_to_scale:
            result[self.features_to_scale] = self.feature_scaler.inverse_transform(
                data[self.features_to_scale],
            )
        return result

    def fit_transform(
        self,
        data: pd.DataFrame,
        target_col: str,
        no_scale_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        self.fit(data, target_col, no_scale_cols)
        return self.transform(data)
