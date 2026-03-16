import hydra
import mlflow
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.data_manager import MultiTickerDataManager
from src.evaluation.backtester import VectorizedBacktester
from src.evaluation.evaluator import ModelEvaluator
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

    data_manager = MultiTickerDataManager(tickers=tickers, sequence_length=sequence_length, batch_size=batch_size)
    train_loader, val_loader = data_manager.build_dataloaders()

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

    with mlflow.start_run():
        trainer.train()

        for ticker, test_loader in data_manager.test_loaders.items():
            preprocessor = data_manager.preprocessors[ticker]
            dates = data_manager.test_dates[ticker]
            close = data_manager.test_close[ticker]

            evaluator = ModelEvaluator(
                ticker=ticker,
                model=model,
                test_loader=test_loader,
                preprocessor=preprocessor,
                target_col="log_return_24",
            )

            test_metrics, y_true, y_pred = evaluator.evaluate()
            print(test_metrics)

            backtester = VectorizedBacktester(ticker, close, dates, y_true, y_pred)
            metrics = backtester.run()
            print(metrics)

            backtester.plot_equity()


if __name__ == "__main__":
    main()
