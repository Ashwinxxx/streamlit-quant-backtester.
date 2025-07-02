This project implements a production-grade quantitative alpha platform with advanced
features for algorithmic trading and portfolio management. It includes a multi-factor alpha
library, machine learning ensemble models, portfolio optimization with risk constraints, and
comprehensive backtesting capabilities.
Features
• Multi-factor Alpha Library: Calculates a wide range of factors including momentum,
value, quality, volatility, and technical indicators.
• ML Ensemble: Utilizes Random Forest and Gradient Boosting models for generating
alpha signals, trained with time-series cross-validation.
• Portfolio Optimization: Advanced optimization with constraints for sector neutrality,
maximum position size, and target volatility.
• Market Impact & Execution Cost Modeling: Estimates transaction costs and market
impact during backtesting.
• Comprehensive Backtesting: Simulates trading strategies over historical data with
detailed performance attribution and risk metrics.
• Flexible Universe: Easily configurable to include a desired number of S&P 500 stocks.
Methods:
• get_sp500_universe() : Returns a list of S&P 500 tickers based on universe_size and their
sector mappings.
• fetch_comprehensive_data(tickers, period="3y") : Fetches historical market data and
fundamental ratios for the given tickers.
• calculate_advanced_factors(data, ticker) : Calculates a comprehensive set of alpha factors
for a given ticker.
• calculate_rsi(prices, window=14) : Helper function to calculate the Relative Strength Index
(RSI).
• prepare_ml_features_advanced(data, tickers) : Prepares the feature set for machine learning
models.
• train_ensemble_model(features_dict, train_end_date, target_horizon='target_21d') : Trains the
Random Forest and Gradient Boosting ensemble models.
• generate_ensemble_predictions(features_dict, prediction_date) : Generates alpha predictions
using the trained ensemble model.
calculate_risk_model(data, tickers, lookback_days=252) : Builds a simplified risk model by
calculating covariance matrix and factor loadings.
• optimize_portfolio(alpha_signals, tickers, max_positions=30) : Performs portfolio optimization
based on alpha signals and risk constraints.
• calculate_market_impact(portfolio_changes, data, tickers) : Estimates market impact and
execution costs for portfolio rebalancing.
• run_production_backtest(start_date='2022-01-01', end_date='2024-06-01') : Runs the full
production-grade backtest and generates performance metrics and plots.
Performance Metrics
Performance Metrics
After running a backtest, the platform.risk_metrics attribute will contain a dictionary of key
performance indicators, including:
• Annualized Return
• Annualized Volatility
• Sharpe Ratio
• Max Drawdown
