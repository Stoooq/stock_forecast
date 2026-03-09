from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from torch.utils.data import DataLoader


class ModelEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        preprocessor: Any,
        target_col: str,
    ):
        self.model = model
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
                np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])
            )
        )

        return {"rmse": rmse, "mae": mae, "directional_accuracy": da}

    def plot_results(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None
    ) -> None:
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label="Actual", color="blue", alpha=0.7)
        plt.plot(
            y_pred,
            label="Predicted",
            color="red",
            alpha=0.7,
            linestyle="--",
        )

        plt.title("Pred vs True")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Wykres zapisany w: {save_path}")
        else:
            plt.show()

        plt.close()

    def evaluate(self) -> dict[str, float]:
        y_true, y_pred = self._generate_predictions()

        # columns = self.preprocessor.scaler.feature_names_in_

        # dummy_df_true = pd.DataFrame(np.zeros((len(y_true_scaled), len(columns))), columns=columns)
        # dummy_df_true[self.target_col] = y_true_scaled
        # y_true_unscaled_df = self.preprocessor.inverse_transform(dummy_df_true)
        # y_true = y_true_unscaled_df[self.target_col].to_numpy()

        # dummy_df_pred = pd.DataFrame(np.zeros((len(y_pred_scaled), len(columns))), columns=columns)
        # dummy_df_pred[self.target_col] = y_pred_scaled
        # y_pred_unscaled_df = self.preprocessor.inverse_transform(dummy_df_pred)
        # y_pred = y_pred_unscaled_df[self.target_col].to_numpy()

        metrics = self.calculate_metrics(y_true, y_pred)

        self.plot_results(y_true, y_pred)

        return metrics
