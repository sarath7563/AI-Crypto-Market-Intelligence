# Crypto Market Intelligence V5.2 — Spot + Smart Money + Prediction

A research-only Streamlit terminal for crypto spot analysis.

## Main features
- Searchable coin catalogue
- Binance Spot candles when a matching market exists
- CoinGecko daily data with CoinPaprika fallback
- TradingView-style candlestick chart
- EMA/SMA 20/50/200, RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic
- Volume, volume ratio and OBV
- Swing structure, BSL/SSL, BOS, CHoCH and FVG detection
- Multi-timeframe view: 1W, 2D, 1D, 4H, 1H
- Spot-only hypothetical setup: entry, invalidation/stop, TP1/TP2/TP3 and R:R
- **Prediction engine:** 1-day, 3-day and 5-day direction probabilities + expected returns + projected prices + model-spread range
- Random Forest + Logistic Regression ensemble
- Chronological holdout metrics and feature importance
- Explicit NO TRADE / WAIT state
- CSV export

## Run
```powershell
python -m pip install -r requirements.txt
streamlit run crypto_market_intelligence_v5_2.py
```

Or double-click `run_v5.bat`.

## Important
Predictions are statistical research outputs, not guaranteed future prices. A projected price is calculated from the model's expected return. The displayed range is a model/residual spread, not a guaranteed confidence interval. Spot setups are hypothetical and no orders are executed.
