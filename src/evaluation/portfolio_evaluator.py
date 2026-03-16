import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from src.evaluation.backtester import VectorizedBacktester
from src.evaluation.evaluator import ModelEvaluator


class PortfolioEvaluator:
    def __init__(self, model, data_manager, target_col):
        self.model = model
        self.data_manager = data_manager
        self.target_col = target_col
        self.detailed_metrics = {}

    def run(self):
        aggregated_metrics = defaultdict(list)

        for ticker, test_loader in self.data_manager.test_loaders.items():
            preprocessor = self.data_manager.preprocessors[ticker]
            dates = self.data_manager.test_dates[ticker]
            close = self.data_manager.test_close[ticker]

            evaluator = ModelEvaluator(
                ticker=ticker,
                model=self.model,
                test_loader=test_loader,
                preprocessor=preprocessor,
                target_col="log_return_24",
            )
            eval_metrics, y_true, y_pred = evaluator.run()

            fig = evaluator.plot_results(y_true, y_pred)
            mlflow.log_figure(fig, f"predictions/{ticker}_pred.png")
            plt.close(fig)

            backtester = VectorizedBacktester(
                ticker=ticker,
                close=close,
                dates=dates,
                predictions=y_pred,
            )
            backtester_metrics = backtester.run()

            fig = backtester.plot_equity()
            mlflow.log_figure(fig, f"equity_curves/{ticker}_equity.png")
            plt.close(fig)

            combined_metrics = {**eval_metrics, **backtester_metrics}
            self.detailed_metrics[ticker] = combined_metrics

            for key, value in combined_metrics.items():
                aggregated_metrics[key].append(value)

        portfolio_metrics = {
            f"portfolio_mean_{key}": np.mean(values)
            for key, values in aggregated_metrics.items()
        }

        return portfolio_metrics, self.detailed_metrics

    def save_detailed_metrics(self, filepath: str):
        with Path(filepath).open("w") as f:
            json.dump(self.detailed_metrics, f, indent=4)
