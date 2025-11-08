import torch
import numpy as np
import matplotlib.pyplot as plt


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
            loss = self.loss_fn(output, y_batch.unsqueeze(1))

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
                loss = self.loss_fn(output, y_batch.unsqueeze(1))

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

    def evaluate(self, test_loader) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                output = self.model(X_batch)
                loss = self.loss_fn(output, y_batch.unsqueeze(1))

                total_loss += loss.item()

                predictions = output.cpu().numpy()
                targets = y_batch.cpu().numpy()

                all_predictions.extend(predictions.flatten())
                all_targets.extend(targets.flatten())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        avg_loss = total_loss / len(test_loader)
        mae = np.mean(np.abs(all_predictions - all_targets))
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))

        return {
            "loss": avg_loss,
            "mae": mae,
            "rmse": rmse,
            "predictions": all_predictions,
            "targets": all_targets
        }
    
    def plot_history(self, save_path=None, figsize=(12, 5)):
        if not self.train_loss_history:
            print("Train history is empty")
            return

        plt.figure(figsize=figsize)
        
        epochs = range(1, len(self.train_loss_history) + 1)
        
        plt.plot(epochs, self.train_loss_history, 'b-', label='Train Loss', linewidth=2)
        
        if self.val_loss_history:
            plt.plot(epochs, self.val_loss_history, 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training History', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        min_train_loss = min(self.train_loss_history)
        min_train_epoch = self.train_loss_history.index(min_train_loss) + 1
        plt.plot(min_train_epoch, min_train_loss, 'bo', markersize=8)
        plt.annotate(f'Min: {min_train_loss:.4f}', 
                    xy=(min_train_epoch, min_train_loss),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.3),
                    fontsize=9)
        
        if self.val_loss_history:
            min_val_loss = min(self.val_loss_history)
            min_val_epoch = self.val_loss_history.index(min_val_loss) + 1
            plt.plot(min_val_epoch, min_val_loss, 'ro', markersize=8)
            plt.annotate(f'Min: {min_val_loss:.4f}', 
                        xy=(min_val_epoch, min_val_loss),
                        xytext=(10, -20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.3),
                        fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        
        plt.show()

    def plot_predictions(
        self, test_loader, scaler=None, save_path=None, figsize=(14, 6), max_samples=None
    ):
        metrics = self.evaluate(test_loader)
        predictions = metrics["predictions"]
        targets = metrics["targets"]

        if scaler is not None:
            num_features = scaler.n_features_in_
            
            pred_full = np.zeros((len(predictions), num_features))
            pred_full[:, 0] = predictions
            predictions_original = scaler.inverse_transform(pred_full)[:, 0]
            
            target_full = np.zeros((len(targets), num_features))
            target_full[:, 0] = targets
            targets_original = scaler.inverse_transform(target_full)[:, 0]
        else:
            predictions_original = predictions
            targets_original = targets

        if max_samples is not None and len(predictions_original) > max_samples:
            predictions_original = predictions_original[:max_samples]
            targets_original = targets_original[:max_samples]

        fig, ax = plt.subplots(figsize=figsize)
        
        indices = range(len(predictions_original))
        
        ax.plot(
            indices,
            targets_original,
            label="Actual Gold Price",
            color="blue",
            linewidth=2,
            alpha=0.8
        )
        
        ax.plot(
            indices,
            predictions_original,
            label="Predicted Price",
            color="red",
            linewidth=2,
            linestyle="--",
            alpha=0.8
        )
        
        ax.set_xlabel("Sample Index", fontsize=12)
        ax.set_ylabel("Gold Price (USD)" if scaler else "Normalized Value", fontsize=12)
        ax.set_title("Gold Price: Actual vs Predicted (Use toolbar to zoom/pan)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Predictions plot saved: {save_path}")
        
        plt.ion()
        plt.show(block=True)