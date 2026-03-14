import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from src.data_loader import YahooFinanceLoader
from src.dataset import TimeSeriesDataset
from src.evaluation.backtester import VectorizedBacktester
from src.evaluation.evaluator import ModelEvaluator
from src.features import DataPreprocessor
from src.models.simple_linear import SimpleLinear
from src.models.simple_lstm import SimpleLSTM
from src.training.trainer import ModelTrainer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print("Running with config: ")
    print(OmegaConf.to_yaml(cfg))

    tickers = list(cfg.data.tickers)
    sequence_length = cfg.data.sequence_length
    batch_size = cfg.data.batch_size

    all_train_datasets = []
    all_val_datasets = []
    all_test_datasets = []

    for ticker in tickers:
        print(f"Processing ticker: {ticker}")

        yf_loader = YahooFinanceLoader(
            config="c",
            tickers=[ticker],
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

        features_df = preprocessor.add_features(raw_df)
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
            sequence_length,
            target_col="log_return_24",
        )
        val_dataset = TimeSeriesDataset(
            val_scaled,
            sequence_length,
            target_col="log_return_24",
        )
        test_dataset = TimeSeriesDataset(
            test_scaled,
            sequence_length,
            target_col="log_return_24",
        )

        all_train_datasets.append(train_dataset)
        all_val_datasets.append(val_dataset)
        all_test_datasets.append(test_dataset)

    global_train_dataset = ConcatDataset(all_train_datasets)
    global_val_dataset = ConcatDataset(all_val_datasets)
    global_test_dataset = ConcatDataset(all_test_datasets)

    print(f"Train dataset size: {len(global_train_dataset)}")

    train_loader = DataLoader(global_train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(all_test_datasets[0], batch_size, shuffle=False)
    val_loader = DataLoader(all_val_datasets[0], batch_size, shuffle=False)
    # val_loader = DataLoader(global_val_dataset, batch_size=64, shuffle=False)
    # test_loader = DataLoader(global_test_dataset, batch_size=64, shuffle=False)

    model = SimpleLSTM(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        output_size=1,
    )
    loss_fn = nn.HuberLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)

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
