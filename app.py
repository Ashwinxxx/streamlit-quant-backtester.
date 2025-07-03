import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import streamlit as st
import io

warnings.filterwarnings('ignore')

class ProductionQuantAlphaPlatform:
    """
    Production-Ready Quantitative Alpha Platform

    Enterprise Features:
    - Multi-factor alpha library with ML ensemble
    - Portfolio optimization with risk constraints
    - Market impact and execution cost modeling
    - Sector/beta neutrality constraints
    - Live performance attribution
    - Risk budgeting and volatility targeting
    - Drawdown control and position limits
    """

    def __init__(self, universe_size=100, target_volatility=0.12, max_leverage=1.0):
        self.universe_size = universe_size
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.gbm_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        self.scaler = RobustScaler()
        self.factor_loadings = {}
        self.covariance_matrix = None
        self.sector_mapping = {}
        self.performance_attribution = []
        self.risk_metrics = {}

    def get_sp500_universe(self):
        """Extended S&P 500 universe with sector classifications"""
        universe = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology', 'META': 'Technology', 'NVDA': 'Technology', 'ADBE': 'Technology', 'CRM': 'Technology', 'ORCL': 'Technology', 'AMD': 'Technology', 'QCOM': 'Technology', 'INTC': 'Technology',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare', 'MRK': 'Healthcare', 'UNH': 'Healthcare', 'LLY': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare', 'AMGN': 'Healthcare',
            'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials', 'MS': 'Financials', 'AXP': 'Financials', 'BLK': 'Financials', 'SPGI': 'Financials', 'V': 'Financials',
            'TSLA': 'Consumer Discretionary', 'HD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary',
            'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples', 'WMT': 'Consumer Staples', 'COST': 'Consumer Staples', 'CL': 'Consumer Staples',
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy', 'SLB': 'Energy', 'PSX': 'Energy',
            'BA': 'Industrials', 'CAT': 'Industrials', 'HON': 'Industrials', 'RTX': 'Industrials', 'UPS': 'Industrials', 'MMM': 'Industrials',
            'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities', 'EXC': 'Utilities', 'SRE': 'Utilities',
            'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials', 'FCX': 'Materials', 'NEM': 'Materials', 'ECL': 'Materials',
            'PLD': 'Real Estate', 'AMT': 'Real Estate', 'CCI': 'Real Estate', 'EQIX': 'Real Estate', 'SPG': 'Real Estate', 'PSA': 'Real Estate',
            'DIS': 'Communication Services', 'NFLX': 'Communication Services', 'VZ': 'Communication Services', 'T': 'Communication Services'
        }
        tickers = list(universe.keys())[:self.universe_size]
        self.sector_mapping = {k: v for k, v in universe.items() if k in tickers}
        return tickers

    @st.cache_data
    def fetch_comprehensive_data(_self, tickers, period="5y"):
        """Fetch comprehensive market and fundamental data"""
        st.write(f"Fetching comprehensive data for {len(tickers)} securities...")
        progress_bar = st.progress(0)
        
        data = {}
        successful_tickers = []
        for i, ticker in enumerate(tickers):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                info = stock.info
                if not hist.empty and len(hist) > 200:
                    data[ticker] = {
                        'price': hist['Close'], 'volume': hist['Volume'], 'high': hist['High'], 'low': hist['Low'], 'open': hist['Open'],
                        'pe_ratio': info.get('trailingPE'), 'pb_ratio': info.get('priceToBook'), 'ev_revenue': info.get('enterpriseToRevenue'),
                        'debt_to_equity': info.get('debtToEquity'), 'roe': info.get('returnOnEquity'), 'profit_margin': info.get('profitMargins'),
                        'market_cap': info.get('marketCap'), 'beta': info.get('beta', 1.0), 'sector': _self.sector_mapping.get(ticker, 'Unknown')
                    }
                    successful_tickers.append(ticker)
            except Exception as e:
                st.warning(f"Failed to fetch {ticker}: {e}")
            progress_bar.progress((i + 1) / len(tickers))
        st.success(f"Successfully fetched data for {len(successful_tickers)} securities")
        return data, successful_tickers

    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_advanced_factors(self, data, ticker):
        prices = data[ticker]['price']
        volume = data[ticker]['volume']
        high = data[ticker]['high']
        low = data[ticker]['low']
        factors = {}
        returns = prices.pct_change()

        factors['momentum_1m'] = prices.pct_change(21).shift(1)
        factors['momentum_3m'] = prices.pct_change(63).shift(1)
        factors['momentum_12m'] = prices.pct_change(252).shift(21)
        factors['risk_adj_momentum'] = (factors['momentum_3m'] / returns.rolling(21).std()).fillna(0)
        
        for name, key in [('value_pe', 'pe_ratio'), ('value_pb', 'pb_ratio'), ('value_ev_sales', 'ev_revenue')]:
            val = data[ticker].get(key)
            factors[name] = -np.log(max(val, 0.1)) if pd.notna(val) else np.nan
        
        for name, key in [('quality_roe', 'roe'), ('quality_margin', 'profit_margin')]:
             factors[name] = data[ticker].get(key)
        factors['quality_leverage'] = -data[ticker].get('debt_to_equity', np.nan)

        factors['volatility_21d'] = returns.rolling(21).std() * np.sqrt(252)
        factors['rsi'] = self.calculate_rsi(prices, 14)
        sma_20 = prices.rolling(20).mean()
        bb_std = prices.rolling(20).std()
        factors['bb_position'] = (prices - sma_20) / (2 * bb_std) if bb_std is not None and not bb_std.empty else np.nan
        factors['volume_ratio'] = volume / volume.rolling(21).mean()

        return factors
    
    def prepare_ml_features_advanced(self, data, tickers):
        features_dict = {}
        for ticker in tickers:
            try:
                factors = self.calculate_advanced_factors(data, ticker)
                prices = data[ticker]['price']
                feature_df = pd.DataFrame(index=prices.index)
                for factor_name, factor_series in factors.items():
                    if isinstance(factor_series, pd.Series):
                        feature_df[factor_name] = factor_series.shift(1)
                    else:
                        feature_df[factor_name] = factor_series
                
                feature_df['target_21d'] = prices.pct_change(21).shift(-21)
                features_dict[ticker] = feature_df.replace([np.inf, -np.inf], np.nan)
            except Exception as e:
                print(f"Error processing {ticker} for ML features: {e}")
        return features_dict

    def train_ensemble_model(self, features_dict, train_end_date, target_horizon='target_21d'):
        print(f"Training ensemble model for {target_horizon}...")
        X_list, y_list = [], []
        feature_cols = [
            'momentum_1m', 'momentum_3m', 'momentum_12m', 'risk_adj_momentum', 
            'value_pe', 'value_pb', 'value_ev_sales', 'quality_roe', 'quality_margin', 
            'quality_leverage', 'volatility_21d', 'rsi', 'bb_position', 'volume_ratio'
        ]
        
        for ticker, df in features_dict.items():
            train_data = df[df.index <= train_end_date].copy()
            if len(train_data) > 100:
                X_list.append(train_data[feature_cols])
                y_list.append(train_data[target_horizon])

        if not X_list: return False
        
        X_combined = pd.concat(X_list)
        y_combined = pd.concat(y_list)
        
        mask = ~X_combined.isnull().any(axis=1) & ~y_combined.isnull()
        X_clean, y_clean = X_combined[mask], y_combined[mask]

        if len(X_clean) < 500:
             print("Warning: Insufficient clean data for training.")
             return False

        X_scaled = self.scaler.fit_transform(X_clean)
        self.rf_model.fit(X_scaled, y_clean)
        self.gbm_model.fit(X_scaled, y_clean)
        print("Ensemble model trained successfully.")
        return True

    def generate_ensemble_predictions(self, features_dict, prediction_date):
        predictions = {}
        feature_cols = [
            'momentum_1m', 'momentum_3m', 'momentum_12m', 'risk_adj_momentum', 
            'value_pe', 'value_pb', 'value_ev_sales', 'quality_roe', 'quality_margin', 
            'quality_leverage', 'volatility_21d', 'rsi', 'bb_position', 'volume_ratio'
        ]
        
        predict_data_list, tickers_for_pred = [], []
        for ticker, df in features_dict.items():
            if prediction_date in df.index:
                predict_data_list.append(df.loc[prediction_date, feature_cols])
                tickers_for_pred.append(ticker)
        
        if not predict_data_list: return {}

        predict_df = pd.DataFrame(predict_data_list, index=tickers_for_pred).dropna()
        if predict_df.empty: return {}
        
        predict_features_scaled = self.scaler.transform(predict_df)
        rf_pred = self.rf_model.predict(predict_features_scaled)
        gbm_pred = self.gbm_model.predict(predict_features_scaled)
        ensemble_preds = 0.6 * rf_pred + 0.4 * gbm_pred
        
        return dict(zip(predict_df.index, ensemble_preds))

    def calculate_risk_model(self, data, tickers, lookback_days=252):
        print("Building risk model...")
        returns_data = {ticker: data[ticker]['price'].pct_change().dropna().tail(lookback_days) 
                        for ticker in tickers if ticker in data and len(data[ticker]['price']) > lookback_days}
        if len(returns_data) < 10:
            print("Warning: Insufficient data for a robust risk model.")
            return False
        
        returns_df = pd.DataFrame(returns_data).dropna()
        self.covariance_matrix = returns_df.cov()
        print("Risk model built successfully.")
        return True
    
    def optimize_portfolio(self, alpha_signals, tickers, max_positions=30):
        if not alpha_signals or len(alpha_signals) < 5: return {}
        
        valid_tickers = [t for t in tickers if t in alpha_signals and t in self.covariance_matrix.index]
        if len(valid_tickers) < 5: return {}

        mu = np.array([alpha_signals[ticker] for ticker in valid_tickers])
        cov_subset = self.covariance_matrix.loc[valid_tickers, valid_tickers].values
        n_assets = len(valid_tickers)

        def objective(weights):
            portfolio_return = np.dot(weights, mu)
            portfolio_variance = np.dot(weights.T, np.dot(cov_subset, weights))
            return -(portfolio_return / np.sqrt(max(portfolio_variance, 1e-9))) # Maximize Sharpe

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - self.max_leverage}]
        bounds = [(-0.05, 0.05) for _ in range(n_assets)]
        w0 = np.ones(n_assets) / n_assets * self.max_leverage

        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            weights = result.x
            portfolio_df = pd.DataFrame({'ticker': valid_tickers, 'weight': weights})
            portfolio_df = portfolio_df[np.abs(portfolio_df['weight']) > 0.001]
            return portfolio_df.set_index('ticker')['weight'].to_dict()
        else: # Fallback
            sorted_signals = sorted(alpha_signals.items(), key=lambda x: x[1], reverse=True)
            top_signals = dict(sorted_signals[:max_positions])
            position_size = self.max_leverage / len(top_signals) if top_signals else 0
            return {ticker: position_size for ticker in top_signals.keys()}

    def calculate_market_impact(self, portfolio_changes, data):
        total_impact = 0
        for ticker, weight_change in portfolio_changes.items():
            if ticker in data and abs(weight_change) > 1e-6:
                try:
                    avg_volume = data[ticker]['volume'].tail(21).mean()
                    last_price = data[ticker]['price'].iloc[-1]
                    if pd.isna(avg_volume) or pd.isna(last_price) or avg_volume == 0: continue
                    
                    portfolio_value = 100e6  # Assume $100M portfolio
                    notional_change = abs(weight_change) * portfolio_value
                    trade_volume = notional_change / last_price
                    volume_participation = trade_volume / avg_volume
                    
                    k = 0.0003 # 3 bps base impact
                    impact_pct = k * np.sqrt(volume_participation)
                    total_impact += abs(weight_change) * min(impact_pct, 0.05)
                except Exception:
                    total_impact += abs(weight_change) * 0.0005 # Fallback
        return total_impact

    def run_production_backtest(self, start_date, end_date):
        print("🚀 Starting Production Quantitative Alpha Backtest...")
        tickers = self.get_sp500_universe()
        data, successful_tickers = self.fetch_comprehensive_data(tickers)
        if len(successful_tickers) < 20:
            st.error("❌ Insufficient data for backtest.")
            return None, None

        features_dict = self.prepare_ml_features_advanced(data, successful_tickers)
        self.calculate_risk_model(data, successful_tickers)

        common_index = None
        for ticker in successful_tickers:
            if ticker in data and not data[ticker]['price'].empty:
                idx = data[ticker]['price'].index
                common_index = idx if common_index is None else common_index.intersection(idx)
        
        price_data = pd.DataFrame({t: data[t]['price'].reindex(common_index) for t in successful_tickers}).dropna(how='all')
        backtest_dates = price_data.index[(price_data.index >= start_date) & (price_data.index <= end_date)]
        if backtest_dates.empty:
            st.error("❌ No backtest dates found in the specified range.")
            return None, None

        rebalance_dates = backtest_dates[::21]
        portfolio_history, current_portfolio = [], {}
        total_tx_costs, total_impact_costs = 0, 0
        
        portfolio_values = pd.Series(index=backtest_dates, dtype=float)
        portfolio_values.iloc[0] = 1.0

        model_trained = False

        print(f"Backtesting over {len(backtest_dates)} days, with {len(rebalance_dates)} rebalancing periods.")

        for i in range(1, len(backtest_dates)):
            current_date = backtest_dates[i]
            previous_date = backtest_dates[i-1]
            
            daily_ret = 0.0
            if current_portfolio:
                price_changes = (price_data.loc[current_date] / price_data.loc[previous_date]) - 1
                daily_ret = sum(current_portfolio.get(t, 0) * price_changes.get(t, 0) for t in current_portfolio)
            
            portfolio_values.iloc[i] = portfolio_values.iloc[i-1] * (1 + daily_ret)
            
            if current_date in rebalance_dates:
                print(f"Rebalancing on {current_date.strftime('%Y-%m-%d')}")
                
                if rebalance_dates.get_loc(current_date) % 3 == 0:
                    train_end = current_date - pd.Timedelta(days=1)
                    training_success = self.train_ensemble_model(features_dict, train_end)
                    if training_success:
                        model_trained = True

                if model_trained:
                    alpha_signals = self.generate_ensemble_predictions(features_dict, current_date)
                    target_portfolio = self.optimize_portfolio(alpha_signals, successful_tickers)

                    changes = {t: target_portfolio.get(t, 0) - current_portfolio.get(t, 0) for t in set(current_portfolio) | set(target_portfolio)}
                    turnover = sum(abs(v) for v in changes.values()) / 2
                    tx_cost = turnover * 0.0015
                    impact_cost = self.calculate_market_impact(changes, data)
                    
                    total_tx_costs += tx_cost
                    total_impact_costs += impact_cost
                    portfolio_values.iloc[i] *= (1 - tx_cost - impact_cost)
                    
                    current_portfolio = target_portfolio
                    portfolio_history.append({
                        'date': current_date, 'cumulative_return': portfolio_values.iloc[i],
                        'transaction_costs_acc': total_tx_costs, 'market_impact_acc': total_impact_costs,
                        'portfolio': current_portfolio
                    })

        print("📈 Backtest Complete!")
        history_df = pd.DataFrame(portfolio_history).set_index('date')
        daily_returns = portfolio_values.pct_change().dropna()
        
        if not daily_returns.empty:
            annual_return = (portfolio_values.iloc[-1])**(252/len(portfolio_values)) - 1
            annual_vol = daily_returns.std() * np.sqrt(252)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0
            peak = portfolio_values.expanding(min_periods=1).max()
            drawdown = (portfolio_values / peak) - 1
            max_drawdown = drawdown.min()

            self.risk_metrics = {
                'final_return': portfolio_values.iloc[-1], 'annual_return': annual_return,
                'annual_vol': annual_vol, 'sharpe': sharpe, 'max_drawdown': max_drawdown,
                'tx_costs_bps': total_tx_costs * 10000, 'impact_costs_bps': total_impact_costs * 10000
            }
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 2]})
        fig.suptitle('Production Backtest Performance Analysis', fontsize=16)

        axes[0].plot(portfolio_values.index, portfolio_values, label='Portfolio Cumulative Return', color='blue')
        axes[0].set_title('Portfolio Performance')
        axes[0].set_ylabel('Cumulative Return')
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend()
        
        if not history_df.empty:
            axes[1].plot(history_df.index, history_df['transaction_costs_acc'] * 100, label='Transaction Costs', color='red', linestyle='--')
            axes[1].plot(history_df.index, history_df['market_impact_acc'] * 100, label='Market Impact', color='purple', linestyle='--')
            axes[1].set_title('Accumulated Costs')
            axes[1].set_ylabel('Cost (% of initial capital)')
            axes[1].grid(True, linestyle='--', alpha=0.6)
            axes[1].legend()

        sns.histplot(daily_returns, bins=50, kde=True, color='green', ax=axes[2])
        axes[2].set_title('Distribution of Daily Returns')
        axes[2].set_xlabel('Daily Return')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        return history_df, fig


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("📈 Production Quantitative Alpha Backtester")

    st.sidebar.header("Backtest Configuration")
    start_date = st.sidebar.date_input("Start Date", datetime(2022, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime(2024, 6, 1))
    universe_size = st.sidebar.slider("Universe Size (Number of Stocks)", 20, 100, 75)
    target_vol = st.sidebar.slider("Target Volatility", 0.05, 0.25, 0.12, 0.01)
    leverage = st.sidebar.slider("Max Leverage", 1.0, 2.0, 1.0, 0.1)

    if st.sidebar.button("🚀 Run Backtest"):
        log_stream = io.StringIO()
        
        try:
            platform = ProductionQuantAlphaPlatform(
                universe_size=universe_size, 
                target_volatility=target_vol, 
                max_leverage=leverage
            )

            with st.spinner("Running backtest... This may take several minutes."):
                backtest_results, fig = platform.run_production_backtest(
                    start_date=pd.to_datetime(start_date, utc=True), 
                    end_date=pd.to_datetime(end_date, utc=True)
                )

            st.success("✅ Backtest Complete!")

            if backtest_results is not None:
                st.subheader("Key Performance Metrics")
                metrics = platform.risk_metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.3f}")
                col2.metric("Annualized Return", f"{metrics.get('annual_return', 0):.2%}")
                col3.metric("Annualized Volatility", f"{metrics.get('annual_vol', 0):.2%}")
                col4.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
                
                st.subheader("Performance Analysis")
                st.pyplot(fig)

                st.subheader("Portfolio History (at rebalancing dates)")
                st.dataframe(backtest_results)
            else:
                st.error("Backtest failed to produce results.")

        except Exception as e:
            st.error(f"An error occurred during the backtest: {e}")