from src.data_loader import YahooFinanceLoader
from src.features import DataPreprocessor


def main():
    yf_loader = YahooFinanceLoader(
        config="cos",
        tickers=["AAPL"],
        start_date="2022-01-01",
        end_date="2025-01-01",
        save_dir="data/raw/",
        interval="1d",
    )

    data = yf_loader.process()

    data_pre = DataPreprocessor(
        features={
            "log_return": {"periods": [1, 2, 3], "column": "close"},
            "rsi": {"periods": [14, 21, 28], "column": "close"},
        }
    )

    df = data_pre.add_features(data)

    print(df)


if __name__ == "__main__":
    main()
