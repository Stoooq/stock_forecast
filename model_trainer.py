import torch


class ModelTrainer:
    def __init__(self, model, loss_fn, optimizer, device="cpu"):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.train_loss_history = []
        self.val_loss_history = []

    def train_epoch(self, train_loader) -> float:
        self.model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            output = self.model(X_batch)
            loss = self.loss_fn(output, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        return avg_loss

    def validate_epoch(self, val_loader) -> float:
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                output = self.model(X_batch)
                loss = self.loss_fn(output, y_batch)

                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(val_loader)
        return avg_loss

    def fit(
        self, train_loader, val_loader=None, epochs: int = 100, verbose: bool = True
    ):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_loss_history.append(train_loss)

            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader)
                self.val_loss_history.append(val_loss)

                if verbose and epoch % 5 == 0:
                    print(
                        f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                    )
            else:
                if verbose and epoch % 5 == 0:
                    print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}")

        if verbose:
            print("\nTraining ended")
            print(f"Final Train Loss: {self.train_loss_history[-1]:.6f}")
            if self.val_loss_history:
                print(f"Final Val Loss: {self.val_loss_history[-1]:.6f}")

        return {
            "train_loss": self.train_loss_history,
            "val_loss": self.val_loss_history if val_loader else None,
        }

    def predict(self, X) -> torch.Tensor:
        self.model.eval()

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        X = X.to(self.device)

        with torch.no_grad():
            predictions = self.model(X)

        return predictions
