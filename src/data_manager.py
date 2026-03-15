import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader

from src.data_loader import YahooFinanceLoader
from src.dataset import TimeSeriesDataset
from src.features import DataPreprocessor


class MultiTickerDataManager:
    def __init__(self, tickers: list[str], sequence_length: int, batch_size: int):
        self.tickers = tickers
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.test_loaders = {}
        self.preprocessors = {}
        self.test_dates = {}

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        all_train_datasets = []
        all_val_datasets = []

        for ticker in self.tickers:
            print(f"Processing ticker: {ticker}")

            df = self._fetch_data(ticker)

            preprocessor = DataPreprocessor(
                features={
                    "log_return": {
                        "periods": [1 * 24, 2 * 24, 3 * 24],
                        "column": "close",
                    },
                    "rsi": {"periods": [14 * 24, 21 * 24, 28 * 24], "column": "close"},
                    "rolling_volatility": {
                        "periods": [14 * 24, 30 * 24],
                        "column": "close",
                    },
                    "momentum": {"periods": [10 * 24, 20 * 24], "column": "close"},
                    "zscore": {"periods": [20 * 24], "column": "close"},
                    "macd": {
                        "column": "close",
                        "fast_period": 12 * 24,
                        "slow_period": 26 * 24,
                        "signal_period": 9 * 24,
                    },
                    "ema": {"periods": [10 * 24, 20 * 24, 50 * 24], "column": "close"},
                    "target_direction": {"column": "close"},
                },
            )

            features_df = preprocessor.add_features(df)
            features_df = features_df.fillna(0.0)

            n = len(features_df)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)

            train_df = features_df.iloc[:train_end]
            val_df = features_df.iloc[train_end:val_end]
            test_df = features_df.iloc[val_end:]

            preprocessor.fit(train_df, target_col="log_return_24")
            train_scaled = preprocessor.transform(train_df)
            val_scaled = preprocessor.transform(val_df)
            test_scaled = preprocessor.transform(test_df)

            train_dataset = TimeSeriesDataset(
                train_scaled,
                self.sequence_length,
                target_col="log_return_24",
            )
            val_dataset = TimeSeriesDataset(
                val_scaled,
                self.sequence_length,
                target_col="log_return_24",
            )
            test_dataset = TimeSeriesDataset(
                test_scaled,
                self.sequence_length,
                target_col="log_return_24",
            )

            all_train_datasets.append(train_dataset)
            all_val_datasets.append(val_dataset)

            self.test_loaders[ticker] = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )
            self.preprocessors[ticker] = preprocessor
            self.test_dates[ticker] = test_df.index[self.sequence_length:]

        global_train_dataset = ConcatDataset(all_train_datasets)
        global_val_dataset = ConcatDataset(all_val_datasets)

        train_loader = DataLoader(global_train_dataset, self.batch_size, shuffle=True)
        val_loader = DataLoader(global_val_dataset, self.batch_size, shuffle=False)

        return train_loader, val_loader

    def _fetch_data(self, ticker: str) -> pd.DataFrame:
        yf_loader = YahooFinanceLoader(
            config="c",
            tickers=[ticker],
            start_date="2024-06-01",
            end_date="2026-06-01",
            save_dir="data/raw/",
            interval="1h",
        )

        raw_df = yf_loader.process()

        return raw_df
