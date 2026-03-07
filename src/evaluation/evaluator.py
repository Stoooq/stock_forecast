import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

class Evaluator:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.available_metrics = {
            "mae": self._mae,
            "rmse": self._rmse,
            "directional_accuracy": self._directional_accuracy
        }
        self.metrics_to_compute = list(self.available_metrics.keys())

    @staticmethod
    def _mae(y_true, y_pred):
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def _rmse(y_true, y_pred):
        return float(root_mean_squared_error(y_true, y_pred))

    @staticmethod
    def _directional_accuracy(y_true, y_pred):
        return float(np.mean(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])))

    def evaluate(self, test_loader):
        self.model.eval()
        y, pred = [], []

        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(self.cfg.device)
            y_batch = y_batch.to(self.cfg.device)

            pred_batch = self.model(X_batch)

            y.append(y_batch.detach().cpu().numpy().reshape(-1))
            pred.append(pred_batch.detach().cpu().numpy().reshape(-1))

        y_true = np.concatenate(y)
        y_pred = np.concatenate(pred)
        
        results = {}
        for name in self.metrics_to_compute:
            if name in self.available_metrics:
                results[name] = self.available_metrics[name](y_true, y_pred)

        return y_true, y_pred, results