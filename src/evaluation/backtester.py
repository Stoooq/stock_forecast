import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class VectorizedBacktester:
    def __init__(
        self,
        ticker,
        close,
        dates: np.ndarray,
        prices: np.ndarray,
        predictions: np.ndarray,
        initial_capital: float = 10000.0,
        trading_fee: float = 0.001,
        position_size: float = 0.5,
    ):
        self.ticker = ticker
        self.df = pd.DataFrame(
            {"price": prices, "prediction": predictions, "close": close},
            index=pd.to_datetime(dates),
        )
        self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.position_size = position_size

    def _generate_signals(
        self, threshold: float = 0.0, short: bool = False
    ) -> pd.Series:
        if short:
            return np.where(
                self.df["prediction"] > threshold,
                1,
                np.where(self.df["prediction"] < -threshold, -1, 0),
            )
        return np.where(self.df["prediction"] > threshold, 1, 0)

    def calculate_metrics(self) -> dict[str, float]:
        strat_returns = self.df["raw_strategy_return"]

        total_return = (
            self.df["strategy_equity"].iloc[-1] / self.initial_capital
        ) - 1.0

        days = len(self.df)
        annualized_return = (1 + total_return) ** (252 / days) - 1.0 if days > 0 else 0

        sharpe_ratio = np.sqrt(252) * (
            strat_returns.mean() / (strat_returns.std() + 1e-9)
        )

        rolling_max = self.df["strategy_equity"].cummax()
        drawdown = (self.df["strategy_equity"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        active_days = strat_returns[strat_returns != 0]
        win_rate = (
            len(active_days[active_days > 0]) / len(active_days)
            if len(active_days) > 0
            else 0
        )

        metrics = {
            "total_return_pct": total_return * 100,
            "annualized_return_pct": annualized_return * 100,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown * 100,
            "win_rate_pct": win_rate * 100,
            "total_trades": int(self.df["trades"].sum()),
        }
        return metrics

    def plot_equity(self):
        plt.figure(figsize=(14, 7))
        plt.plot(
            self.df.index,
            self.df["strategy_equity"],
            label="Model Strategy",
            color="green",
            linewidth=2,
        )
        plt.plot(
            self.df.index,
            self.df["market_equity"],
            label="Buy & Hold",
            color="gray",
            alpha=0.6,
        )

        plt.title(f"Algorithmic Strategy vs Buy & Hold ({self.ticker})")
        plt.xlabel("Date")
        plt.ylabel("Capital ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def run(self) -> dict[str, float]:
        self.df["signal"] = self._generate_signals(threshold=0.0, short=False)
        self.df["position"] = self.df["signal"].shift(1).fillna(0)

        self.df["market_return"] = self.df["close"].pct_change().fillna(0.0)

        self.df["raw_strategy_return"] = self.df["position"] * self.df["market_return"]

        self.df["trades"] = self.df["position"].diff().abs().fillna(0.0)
        self.df["raw_strategy_return"] -= self.df["trades"] * self.trading_fee

        self.df["portfolio_return"] = (
            self.df["raw_strategy_return"] * self.position_size
        )

        daily_multiplier = np.maximum(1.0 + self.df["portfolio_return"], 0.0)
        self.df["strategy_equity"] = self.initial_capital * daily_multiplier.cumprod()

        bh_multiplier = np.maximum(1.0 + self.df["market_return"], 0.0)
        self.df["market_equity"] = self.initial_capital * bh_multiplier.cumprod()

        return self.calculate_metrics()
