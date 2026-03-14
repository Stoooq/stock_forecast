import mlflow
import torch


class ModelTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = criterion
        self.optim = optimizer
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _train_epoch(self) -> float:
        self.model.train()
        epoch_loss = 0.0

        for X, y in self.train_loader:
            X_batch = X.to(self.device)
            y_batch = y.to(self.device)

            output = self.model(X_batch)

            loss = self.loss_fn(output, y_batch.unsqueeze(1))

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss

    def _validate_epoch(self) -> float:
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for X, y in self.val_loader:
                X_batch = X.to(self.device)
                y_batch = y.to(self.device)

                output = self.model(X_batch)
                loss = self.loss_fn(output, y_batch.unsqueeze(1))

                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        params = {
            "epochs": 10,
            "learning_rate": 1e-2,
            "batch_size": 64,
        }
        with mlflow.start_run():
            mlflow.log_params(params)

            for epoch in range(params["epochs"]):
                train_loss = self._train_epoch()
                print(train_loss)

                val_loss = self._validate_epoch()
                print(val_loss)

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

            mlflow.pytorch.log_model(self.model, name="model")
