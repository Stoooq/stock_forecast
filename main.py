import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_loader import YahooFinanceLoader
from src.dataset import TimeSeriesDataset
from src.evaluation.backtester import VectorizedBacktester
from src.evaluation.evaluator import ModelEvaluator
from src.features import DataPreprocessor
from src.models.simple_linear import SimpleLinear
from src.models.simple_lstm import SimpleLSTM
from src.training.trainer import ModelTrainer


def main():
    yf_loader = YahooFinanceLoader(
        config="c",
        tickers=["AAPL"],
        start_date="2024-06-01",
        end_date="2026-06-01",
        save_dir="data/raw/",
        interval="1h",
    )

    raw_df = yf_loader.process()

    preprocessor = DataPreprocessor(
        features={
            "log_return": {"periods": [1 * 24, 2 * 24, 3 * 24], "column": "close"},
            "rsi": {"periods": [14 * 24, 21 * 24, 28 * 24], "column": "close"},
            "rolling_volatility": {"periods": [14 * 24, 30 * 24], "column": "close"},
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

    features_df = preprocessor.add_features(raw_df)

    print(features_df.isnull().sum())

    features_df = features_df.fillna(0.0)

    print(features_df)

    n = len(features_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = features_df.iloc[:train_end]
    val_df = features_df.iloc[train_end:val_end]
    test_df = features_df.iloc[val_end:]

    print(train_df)

    preprocessor.fit(train_df, target_col="target_direction")
    train_scaled = preprocessor.transform(train_df)
    val_scaled = preprocessor.transform(val_df)
    test_scaled = preprocessor.transform(test_df)

    print(train_scaled)

    train_dataset = TimeSeriesDataset(
        train_scaled,
        sequence_length=14,
        target_col="log_return_24",
    )
    val_dataset = TimeSeriesDataset(
        val_scaled,
        sequence_length=14,
        target_col="log_return_24",
    )
    test_dataset = TimeSeriesDataset(
        test_scaled,
        sequence_length=14,
        target_col="log_return_24",
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SimpleLSTM(input_size=23, hidden_size=128, num_layers=2, output_size=1)
    loss_fn = nn.HuberLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=0.0001)

    # model = SimpleLinear(input_size=308, hidden_size=16, output_size=1)
    # loss_fn = nn.MSELoss()
    # optim = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = ModelTrainer(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optim,
        "c",
    )

    trainer.train()

    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        preprocessor=preprocessor,
        target_col="log_return_24",
    )

    test_metrics, y_true, y_pred = evaluator.evaluate()

    print(test_metrics)

    dates = test_df.index[14:]

    backtester = VectorizedBacktester(dates, y_true, y_pred)

    metrics = backtester.run()

    print(metrics)

    backtester.plot_equity()


if __name__ == "__main__":
    main()
