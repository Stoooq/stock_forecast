import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_loader import YahooFinanceLoader
from src.dataset import TimeSeriesDataset
from src.evaluation.model_evaluator import ModelEvaluator
from src.features import DataPreprocessor
from src.models.simple_lstm import SimpleLSTM
from src.training.model_trainer import ModelTrainer


def main():
    yf_loader = YahooFinanceLoader(
        config="c",
        tickers=["AAPL"],
        start_date="2022-01-01",
        end_date="2025-01-01",
        save_dir="data/raw/",
        interval="1d",
    )

    raw_df = yf_loader.process()

    preprocessor = DataPreprocessor(
        features={
            "log_return": {"periods": [1, 2, 3], "column": "close"},
            "rsi": {"periods": [14, 21, 28], "column": "close"},
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

    print(train_df)

    preprocessor.fit(train_df, target_col="target_direction")
    train_scaled = preprocessor.transform(train_df)
    val_scaled = preprocessor.transform(val_df)
    test_scaled = preprocessor.transform(test_df)

    print(train_scaled)

    train_dataset = TimeSeriesDataset(
        train_scaled,
        sequence_length=14,
        target_col="target_direction",
    )
    val_dataset = TimeSeriesDataset(
        val_scaled,
        sequence_length=14,
        target_col="target_direction",
    )
    test_dataset = TimeSeriesDataset(
        test_scaled,
        sequence_length=14,
        target_col="target_direction",
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SimpleLSTM(input_size=14, hidden_size=256, num_layers=5, output_size=1)
    loss_fn = nn.HuberLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)

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
        target_col="target_direction",
    )

    test_metrics = evaluator.evaluate()

    print(test_metrics)


if __name__ == "__main__":
    main()
