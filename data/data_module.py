import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader
from .dataset import TimeSeriesDataset


class DataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scaler = None

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.csv_path, parse_dates=True)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["YesterdayClose"] = df["Close"].shift(1)
        df["YesterdayOpenLogR"] = np.log(df["Open"] / df["Open"].shift(1)).shift(1)
        df["YesterdayHighLogR"] = np.log(df["High"] / df["High"].shift(1)).shift(1)
        df["YesterdayLowLogR"] = np.log(df["Low"] / df["Low"].shift(1)).shift(1)
        df["YesterdayVolumeLogR"] = np.log((df["Volume"] + 1) / (df["Volume"].shift(1) + 1)).shift(1)
        df["YesterdayCloseLogR"] = np.log(df["Close"] / df["Close"].shift(1)).shift(1)

        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["log_volume"] = np.log(df["Volume"] + 1)

        for window in [10, 20, 30]:
            df[f"MA{window}"] = df["Close"].rolling(window=window).mean()
        
        df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
        df["EMA30"] = df["Close"].ewm(span=30, adjust=False).mean()

        if "Date" in df.columns:
            dates = pd.to_datetime(df["Date"])
            df["DayOfWeek"] = dates.dt.dayofweek
            df["DayOfMonth"] = dates.dt.day
            df["MonthNumber"] = dates.dt.month
        else:
            dates = pd.to_datetime(df.index)
            df["DayOfWeek"] = dates.dayofweek
            df["DayOfMonth"] = dates.day
            df["MonthNumber"] = dates.month

        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df["RSI"] = 100 - (100 / (1 + rs))

        exp12 = df["Close"].ewm(span=12, adjust=False).mean()
        exp26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp12 - exp26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        sma20 = df["Close"].rolling(window=20).mean()
        std20 = df["Close"].rolling(window=20).std()
        df["BollingerUpper"] = sma20 + (std20 * 2)
        df["BollingerLower"] = sma20 - (std20 * 2)

        df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        for window in [10, 20, 30]:
            df[f"Volatility_{window}"] = log_ret.rolling(window=window).std()

        # df["volatility_5d"] = log_ret.rolling(window=5).std()
        # df["volatility_20d"] = log_ret.rolling(window=20).std()

        # df["momentum_5d"] = df["Close"] / df["Close"].shift(5) - 1
        # df["momentum_20d"] = df["Close"] / df["Close"].shift(20) - 1
        
        # df["skew_5d"] = log_ret.rolling(window=5).skew()
        
        # df["ZScore"] = (df["Close"] - sma20) / (std20 + 1e-10)

        # df["overnight_gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
        
        # df["abnormal_vol"] = df["Volume"] / (df["Volume"].rolling(window=20).mean() + 1)

        external_cols = [
            "insider_shares", "insider_amount", "insider_buy_flag", 
            "sentiment", "num_articles"
        ]
        for col in external_cols:
            if col not in df.columns:
                df[col] = 0.0

        df["high_low_ratio"] = df["High"] / df["Low"]
        df["close_open_ratio"] = df["Close"] / df["Open"]
        df["price_range"] = (df["High"] - df["Low"]) / df["Close"]
        df["body_ratio"] = abs(df["Close"] - df["Open"]) / (df["High"] - df["Low"] + 1e-10)

        for window in [5, 10, 20, 50]:
            df[f"sma_{window}"] = df["Close"].rolling(window=window).mean()
            df[f"price_to_sma_{window}"] = df["Close"] / df[f"sma_{window}"]

        df = df.dropna()
        
        return df
    
    def get_dates(self, df: pd.DataFrame):
        data_df = df["Date"]
        return data_df[self.cfg.window_size:]

    def create_sequence(
        self, data: np.ndarray, target_col_idx: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []

        for i in range(len(data) - self.cfg.window_size):
            X.append(data[i : i + self.cfg.window_size])
            y.append(data[i + self.cfg.window_size, target_col_idx])

        return np.array(X), np.array(y)

    def fit_scaler(self, data: np.ndarray) -> None:
        if self.cfg.scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif self.cfg.scaler_type == "standard":
            self.scaler = StandardScaler()

        self.scaler.fit(data)

    def transform_data(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)

    def get_scaler(self):
        return self.scaler

    def split_data(
        self, df: pd.DataFrame, val_size: int, test_size: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        total_samples = len(df)

        test_samples = int(total_samples * test_size)
        val_samples = int(total_samples * val_size)
        train_samples = total_samples - test_samples - val_samples

        train_end = train_samples
        val_start = train_end
        val_end = val_start + val_samples
        test_start = val_end

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:val_end].copy()
        test_df = df.iloc[test_start:].copy()

        return train_df, val_df, test_df

    def get_loaders(
        self, val_size: int = 0.1, test_size: int = 0.1
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        df = self.load_data()
        df = self.prepare_features(df)
        df = self.clean_data(df)

        train_df, val_df, test_df = self.split_data(df, val_size, test_size)

        self.feature_cols = [col for col in df if col not in (self.cfg.exclude_cols or [])]
        self.scale_cols = [
            col for col in self.feature_cols if col not in (self.cfg.no_scale_cols or [])
        ]

        test_dates = self.get_dates(test_df)

        self.fit_scaler(train_df[self.scale_cols].values)

        def prepare_array(dataframe):
            data_map = {}
            
            scaled_data = self.transform_data(dataframe[self.scale_cols].values)
            for i, col in enumerate(self.scale_cols):
                data_map[col] = scaled_data[:, i]

            for col in self.cfg.no_scale_cols:
                data_map[col] = dataframe[col].values

            final_data = []
            for col in self.feature_cols:
                final_data.append(data_map[col].reshape(-1, 1))

            return np.hstack(final_data)

        train_arr = prepare_array(train_df)
        val_arr = prepare_array(val_df)
        test_arr = prepare_array(test_df)

        target_idx = self.feature_cols.index(self.cfg.target_col)

        X_train, y_train = self.create_sequence(train_arr, target_col_idx=target_idx)
        X_val, y_val = self.create_sequence(val_arr, target_col_idx=target_idx)
        X_test, y_test = self.create_sequence(test_arr, target_col_idx=target_idx)

        train_ds = TimeSeriesDataset(X_train, y_train)
        val_ds = TimeSeriesDataset(X_val, y_val)
        test_ds = TimeSeriesDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=self.cfg.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, self.scaler, test_dates
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Odwraca skalowanie tylko dla kolumny target (np. log_return).
        Obsługuje sytuację, gdy scaler oczekuje wielu kolumn.
        """
        # Jeśli target nie był skalowany (jest w no_scale_cols), zwróć bez zmian
        if self.cfg.target_col in (self.cfg.no_scale_cols or []):
            return y_scaled

        if self.scaler is None:
            return y_scaled

        # 1. Przygotuj pustą macierz o kształcie, którego oczekuje scaler
        # (n_samples, n_features_in_scaler)
        n_features = self.scaler.n_features_in_
        dummy_matrix = np.zeros((len(y_scaled), n_features))

        # 2. Znajdź indeks targetu wewnątrz kolumn SKALOWANYCH
        # (Uwaga: to nie to samo co feature_cols, bo scaler widzi tylko scale_cols)
        if self.cfg.target_col not in self.scale_cols:
            raise ValueError(f"Target {self.cfg.target_col} not found in scaled columns")
            
        target_idx_in_scaler = self.scale_cols.index(self.cfg.target_col)

        # 3. Wstaw predykcje w odpowiednią kolumnę
        dummy_matrix[:, target_idx_in_scaler] = y_scaled.flatten()

        # 4. Wykonaj inverse transform
        inversed_matrix = self.scaler.inverse_transform(dummy_matrix)

        # 5. Wyciągnij tylko interesującą nas kolumnę
        return inversed_matrix[:, target_idx_in_scaler]
