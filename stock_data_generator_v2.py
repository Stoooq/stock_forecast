from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class StockDataGenerator(Dataset):
    def __init__(
        self,
        csv_path: str,
        target_col: str,
        window_size: int = 30,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        target_type: Literal["log_return", "price", "log_price"] = "log_return",
        split: Literal["train", "val", "test"] = "train",
        scaler_type: Literal["minmax", "standard"] = "minmax",
        scaler=None,
        exclude_cols: list[str] | None = None,
        no_scale_cols: list[str] | None = None,
    ):
        self.csv_path = csv_path
        self.target_col = target_col
        self.window_size = window_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.target_type = target_type
        self.split = split
        self.scaler_type = scaler_type
        self.scaler = scaler
        self.exclude_cols = exclude_cols
        self.no_scale_cols = no_scale_cols

        self.X = None
        self.y = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.ascontiguousarray(self.X[idx], dtype=np.float32),
            np.ascontiguousarray(self.y[idx], dtype=np.float32),
        )

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path, parse_dates=["Date"])
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["log_volume"] = np.log(df["Volume"] + 1)

        df["high_low_ratio"] = df["High"] / df["Low"]
        df["close_open_ratio"] = df["Close"] / df["Open"]
        df["price_range"] = (df["High"] - df["Low"]) / df["Close"]
        df["body_ratio"] = abs(df["Close"] - df["Open"]) / (
            df["High"] - df["Low"] + 1e-10
        )

        for window in [5, 10, 20, 50]:
            df[f"sma_{window}"] = df["Close"].rolling(window=window).mean()
            df[f"price_to_sma_{window}"] = df["Close"] / df[f"sma_{window}"]

        return df

    def _split_data_by_type(self, df: pd.DataFrame) -> pd.DataFrame:
        total_samples = len(df)

        test_samples = int(total_samples * self.test_size)
        val_samples = int(total_samples * self.validation_size)
        train_samples = total_samples - test_samples - val_samples

        train_end = train_samples
        val_start = train_end
        val_end = val_start + val_samples
        test_start = val_end

        if self.split == "train":
            return df.iloc[:train_end].copy()
        elif self.split == "val":
            return df.iloc[val_start:val_end].copy()
        elif self.split == "test":
            return df.iloc[test_start:].copy()

    def _fit_scaler(self, data: np.ndarray) -> None:
        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()

        self.scaler.fit(data)

    def _transform_data(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(data)

    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)

    def _get_scaler(self):
        return self.scaler

    def _create_sequence(
        self, data: np.ndarray, target_column_index: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []

        for i in range(len(data) - self.window_size):
            X.append(data[i : i + self.window_size])
            y.append(data[i + self.window_size, target_column_index])

        return np.array(X), np.array(y)

    def prepare_data(self) -> None:
        df = self._load_data()
        df = self._prepare_features(df)
        df = self._clean_data(df)

        df_split = self._split_data_by_type(df)

        feature_cols = [col for col in df_split if col not in (self.exclude_cols or [])]
        scale_cols = [
            col for col in feature_cols if col not in (self.no_scale_cols or [])
        ]

        if self.split == "train":
            self._fit_scaler(df_split[scale_cols].values)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not provided for non-train split")

        df_split[scale_cols] = self._transform_data(df_split[scale_cols].values).astype(
            np.float32
        )

        # df_scaled = self._transform_data(df_split[scale_cols].values)
        # for i, col in enumerate(scale_cols):
        #     df_split[col] = df_scaled[:, i].astype(np.float32)

        arr = df_split[feature_cols].to_numpy(dtype=np.float32)
        target_idx = feature_cols.index(self.target_col)

        X, y = self._create_sequence(arr, target_column_index=target_idx)

        self.X, self.y = X, y

    def create_dataloader(
        self,
        batch_size: int = 64,
        shuffle: bool = False,
    ) -> DataLoader:
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    @staticmethod
    def create_loaders(
        csv_path: str,
        target_col: str,
        window_size: int = 30,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        target_type: Literal["log_return", "price", "log_price"] = "log_return",
        scaler_type: str = "minmax",
        exclude_cols: list[str] | None = None,
        no_scale_cols: list[str] | None = None,
        batch_size: int = 64,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_gen = StockDataGenerator(
            csv_path=csv_path,
            target_col=target_col,
            window_size=window_size,
            test_size=test_size,
            validation_size=validation_size,
            target_type=target_type,
            split="train",
            scaler_type=scaler_type,
            exclude_cols=exclude_cols,
            no_scale_cols=no_scale_cols,
        )
        train_gen.prepare_data()
        train_loader = train_gen.create_dataloader(batch_size=batch_size, shuffle=True)

        val_gen = StockDataGenerator(
            csv_path=csv_path,
            target_col=target_col,
            window_size=window_size,
            test_size=test_size,
            validation_size=validation_size,
            target_type=target_type,
            split="val",
            scaler=train_gen._get_scaler(),
            exclude_cols=exclude_cols,
            no_scale_cols=no_scale_cols,
        )
        val_gen.prepare_data()
        val_loader = val_gen.create_dataloader(batch_size=batch_size, shuffle=False)

        test_gen = StockDataGenerator(
            csv_path=csv_path,
            target_col=target_col,
            window_size=window_size,
            test_size=test_size,
            validation_size=validation_size,
            target_type=target_type,
            split="test",
            scaler=train_gen._get_scaler(),
            exclude_cols=exclude_cols,
            no_scale_cols=no_scale_cols,
        )
        test_gen.prepare_data()
        test_loader = test_gen.create_dataloader(batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
