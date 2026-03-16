from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch.utils.data import DataLoader


class ModelEvaluator:
    def __init__(
        self,
        ticker: str,
        model: torch.nn.Module,
        test_loader: DataLoader,
        preprocessor: Any,
        target_col: str,
    ):
        self.model = model
        self.ticker = ticker
        self.test_loader = test_loader
        self.preprocessor = preprocessor
        self.target_col = target_col
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _generate_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_y_true = []
        all_y_pred = []

        with torch.no_grad():
            for X, y in self.test_loader:
                X_batch = X.to(self.device)
                y_batch = y.to(self.device)

                output = self.model(X_batch)

                all_y_true.append(y_batch.cpu().numpy().reshape(-1))
                all_y_pred.append(output.cpu().numpy().reshape(-1))

        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)

        return y_true, y_pred

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        rmse = float(root_mean_squared_error(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        da = float(
            np.mean(
                np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1]),
            ),
        )

        return {"rmse": rmse, "mae": mae, "directional_accuracy": da}

    def plot_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        fig = plt.figure(figsize=(14, 7))
        plt.plot(y_true, label="Actual", color="blue", alpha=0.7)
        plt.plot(
            y_pred,
            label="Predicted",
            color="red",
            alpha=0.7,
            linestyle="--",
        )

        plt.title(f"Pred vs True ({self.ticker})")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    def run(self):
        y_true, y_pred = self._generate_predictions()

        y_true_unscaled_df = self.preprocessor.inverse_transform_target(y_true)
        y_pred_unscaled_df = self.preprocessor.inverse_transform_target(y_pred)

        metrics = self.calculate_metrics(y_true_unscaled_df, y_pred_unscaled_df)

        # self.plot_results(y_true_unscaled_df, y_pred_unscaled_df)

        return metrics, y_true_unscaled_df, y_pred_unscaled_df
