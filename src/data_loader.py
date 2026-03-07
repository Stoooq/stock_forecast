from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import yfinance as yf


class BaseDataLoader(ABC):
    def __init__(
        self,
        config,
        tickers: list[str],
        start_date: str,
        end_date: str,
        save_dir: Path | str,
    ):
        self.config = config
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.save_dir = save_dir

    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        return

    @abstractmethod
    def standardize_format(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def save_to_disk(self, data: pd.DataFrame, filename: str) -> Path:
        save_path = self.save_dir + filename
        data.to_csv(save_path)

    def process(self) -> pd.DataFrame:
        data = self.fetch_data()
        standardized_data = self.standardize_format(data)
        filename = "example.csv"
        self.save_to_disk(standardized_data, filename)

        return standardized_data


class YahooFinanceLoader(BaseDataLoader):
    def __init__(
        self,
        config,
        tickers,
        start_date,
        end_date,
        save_dir,
        interval: str,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        super().__init__(config, tickers, start_date, end_date, save_dir)
        self.interval = interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        ##TODO add hydra lib for config

    def fetch_data(self) -> pd.DataFrame:
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date)

        ##TODO add retry to download data from yfinance

        return data

    def standardize_format(self, raw_data) -> pd.DataFrame:
        df = raw_data.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.reset_index()

        df.columns = [str(col).lower() for col in df.columns]

        desired_columns = ["date", "open", "high", "low", "close", "volume"]
        df = df[desired_columns]

        return df

    def save_to_disk(self, data, filename):
        return super().save_to_disk(data, filename)

    def process(self):
        return super().process()
