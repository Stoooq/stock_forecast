# Trading ML Pipeline: End-to-End Deep Learning System

Designed with scalability and professional MLOps standards in mind, this project is an end-to-end Machine Learning pipeline for predicting stock price directional movements. System validates models based on real-world business utility using a custom vectorized backtester that calculates key financial metrics (Sharpe Ratio, Max Drawdown).

## MLflow Tracking & Backtest Showcase

Every training run automatically logs parameters, aggregate portfolio metrics, detailed JSON artifacts, and equity curve plots directly to the MLflow server.
<p align="center">
  <video src="https://github.com/user-attachments/assets/17f122b6-2292-4efa-be4a-6a77a00618a9.mov" width="800" controls="controls"></video>
</p>

## Tech Stack & Tools
* **Deep Learning:** PyTorch
* **Data & Feature Engineering:** Pandas, NumPy, Scikit-learn, yfinance
* **MLOps & Tracking:** MLflow
* **Financial Evaluation:** Custom Vectorized Backtester (Pandas)
* **Environment Management:** uv

## System Architecture & Data Flow

The project is built using Object-Oriented Programming (OOP) principles, ensuring that each module (Preprocessor, Trainer, Backtester) can serve as an independent microservice or integrate seamlessly into a larger quantitative fund infrastructure.

<p align="center">
  <img width="800" alt="MLOps Data Pipeline-2026-03-17-215058" src="https://github.com/user-attachments/assets/bb1ce56c-90a8-402f-abe9-58b038aba599" />
</p>

## Key Capabilities

* **Multi-Ticker Deep Learning:** Trains a single global model across concatenated datasets to learn universal market regimes and prevent single-asset overfitting.
* **Dynamic Configuration (Hydra):** Zero-code hyperparameter tuning directly via CLI for rapid experiment iteration.
* **Professional MLOps (MLflow):** "Aggregation + Artifacts" pattern for clean portfolio-level tracking alongside granular, per-ticker JSON metrics and RAM-generated equity plots.
* **Strict Zero-Leakage Pipeline:** Independent, chronological per-ticker scaling to guarantee absolute absence of forward-looking bias.
* **Vectorized Financial Backtesting:** Fast, loop-free trading simulation factoring in transaction fees and benchmarking against a Buy & Hold strategy.
