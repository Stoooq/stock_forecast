import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from typing import Optional


class DataPreparator:
    def __init__(
        self, data_url: str, window_size: int, test_size: float, validation_size: float
    ):
        self.data_url = data_url
        self.window_size = window_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.scaler = None

    def load_data(
        self,
    ) -> pd.DataFrame:  # TODO: Make class for loading data with diffrent data types
        df = pd.read_csv(self.data_url, parse_dates=["Date"])
        return df

    def clean_data(
        self, df: pd.DataFrame, columns: Optional[list[str]]
    ) -> pd.DataFrame:
        if columns:
            df = df.drop(columns=columns)

        row_NaN = df.isna().any(axis=1).sum()
        if row_NaN < 0.1 * len(df):
            df = df.dropna(axis=0)

        return df

    def fit_scaler(self, data: np.ndarray, scaler_type: str = "minmax") -> None:
        if scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError("scaler_type must be 'minmax' or 'standard'")

        self.scaler.fit(data)

    def transform_data(self, data: np.ndarray) -> np.ndarray:
        data_scaled = self.scaler.transform(data)
        return data_scaled

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        data_unscaled = self.scaler.inverse_transform(data)
        return data_unscaled

    def create_sequence(
        self, data: np.ndarray, target_column_index: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i : i + self.window_size])
            y.append(data[i : i + self.window_size, target_column_index])
        return np.array(X), np.array(y)

    def split_train_test(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        test_samples = int(len(X) * self.test_size)

        X_train = X[:-test_samples]
        X_test = X[-test_samples:]
        y_train = y[:-test_samples]
        y_test = y[-test_samples:]

        return X_train, X_test, y_train, y_test

    def split_train_val_test(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        total_samples = len(X)

        test_samples = int(total_samples * self.test_size)
        val_samples = int(total_samples * self.validation_size)

        X_test = X[-test_samples:]
        y_test = y[-test_samples:]

        X_val = X[-(test_samples + val_samples) : -test_samples]
        y_val = y[-(test_samples + val_samples) : -test_samples]

        X_train = X[: -(test_samples + val_samples)]
        y_train = y[: -(test_samples + val_samples)]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def to_tensor(
        self, *arrays: np.ndarray, dtype: torch.dtype = torch.float32
    ) -> list[torch.Tensor]:
        return [torch.tensor(arr, dtype=dtype) for arr in arrays]

    def create_dataloader(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> DataLoader:
        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader
