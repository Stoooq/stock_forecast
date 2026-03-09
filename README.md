# Trading ML Pipeline: End-to-End Deep Learning System

> **Project Status:** 🚧 Work in Progress (WIP)

Designed with scalability and professional MLOps standards in mind, this project is an end-to-end Machine Learning pipeline for predicting stock price directional movements. Rather than focusing solely on traditional error metrics (like MSE or Accuracy), the system validates models based on real-world business utility using a custom vectorized backtester that calculates key financial metrics (Sharpe Ratio, Max Drawdown).

The project is built using Object-Oriented Programming (OOP) principles, ensuring that each module (Preprocessor, Trainer, Backtester) can serve as an independent microservice or integrate seamlessly into a larger quantitative fund infrastructure.

## Tech Stack & Tools
* **Deep Learning:** PyTorch
* **Data & Feature Engineering:** Pandas, NumPy, Scikit-learn, yfinance
* **MLOps & Tracking:** MLflow
* **Financial Evaluation:** Custom Vectorized Backtester (Pandas)
* **Environment Management:** uv

## System Architecture

The system is divided into fully decoupled, interchangeable modules following the Single Responsibility Principle:

1. **Data Ingestion & Preprocessing (`DataPreprocessor`):** Fetches raw financial data (OHLCV) and standardizes the format.
2. **Feature Engineering:** Generates advanced features and labels.
3. **Model Training (`Trainer`):** Trains Deep Neural Networks in PyTorch utilizing GPU/MPS acceleration.
4. **Evaluation (`ModelEvaluator`):** Translates normalized model signals back into a business context and calculates *Directional Accuracy*.
5. **Strategy Simulation (`VectorizedBacktester`):** Fast, loop-free (vectorized) backtester simulating Position Sizing and trading costs ultimately plotting the strategy's equity curve.

## Experiment Tracking

The project is fully integrated with **MLflow**. Every training run automatically logs:
* Architecture and training hyperparameters.
* Technical metrics (Train/Val Loss, MAE, RMSE).
* Business metrics from the Backtester (Total Return, Annualized Return, Sharpe Ratio, Max Drawdown).
* Visual artifacts (Equity curve, Prediction vs. Actual charts).

This allows for automated searching, comparing, and tracking of the most profitable model configurations over time.

## Roadmap

The system is being built iteratively. Below is the current development status:

### Phase 1: MVP & Core Pipeline (Completed)
- [x] Build data fetching and cleaning mechanism (Data Preprocessor).
- [x] Create and train the baseline PyTorch LSTM model.
- [x] Implement safe data scaling (StandardScaler) and inverse scaling logic.
- [x] Build the Evaluator to generate Prediction vs. Actual charts.
- [x] Develop a Vectorized Backtester simulating portfolio capital, transaction fees, and calculating ROI, Sharpe Ratio, and Max Drawdown.

### Phase 2: Scalability & Advanced Quant Modeling (In Progress)
- [ ] **Global Model Training (Multi-Ticker):** Transition from single-stock training to a global model predicting broad market behavior. Utilize PyTorch's `ConcatDataset` to train the model in parallel on a panel data of 100+ stocks (e.g., S&P 500 constituents) to prevent single-asset overfitting.
- [ ] **Configuration Management:** Integrate **Hydra** (`.yaml` files) for hyperparameter management to instantly launch diverse experiments via CLI without modifying the Python codebase.
- [ ] **Model Research:** Research and implement modern Time-Series architectures (e.g., Temporal Convolutional Networks - TCN, Attention mechanisms / Time-Series Transformers) to significantly improve Directional Accuracy.
- [ ] **Quant Feature Engineering:** Implement industry-standard, non-linear quantitative indicators (Garman-Klass Volatility, Rolling Skewness, Kurtosis, Hurst Exponent).
