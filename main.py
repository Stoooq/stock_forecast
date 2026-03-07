from src.data_loader import YahooFinanceLoader


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

    print(data)


if __name__ == "__main__":
    main()
