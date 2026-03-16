import hydra
import mlflow
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.data_manager import MultiTickerDataManager
from src.evaluation.portfolio_evaluator import PortfolioEvaluator
from src.models.simple_linear import SimpleLinear
from src.models.simple_lstm import SimpleLSTM
from src.training.trainer import ModelTrainer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print("Running with config: ")
    print(OmegaConf.to_yaml(cfg))

    data_manager = MultiTickerDataManager(
        tickers=list(cfg.data.tickers),
        sequence_length=cfg.data.sequence_length,
        batch_size=cfg.data.batch_size,
    )
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
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_fn,
        optimizer=optim,
        epochs=cfg.training.epochs,
    )

    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(cfg.training, resolve=True))
        mlflow.log_params(OmegaConf.to_container(cfg.model, resolve=True))

        trainer.train()

        portfolio_evaluator = PortfolioEvaluator(
            model=model,
            data_manager=data_manager,
            target_col="log_return_24",
        )

        portfolio_metrics, _ = portfolio_evaluator.run()

        mlflow.log_metrics(portfolio_metrics)
        portfolio_evaluator.save_detailed_metrics("detailed_ticker_metrics.json")
        mlflow.log_artifact("detailed_ticker_metrics.json")


if __name__ == "__main__":
    main()
