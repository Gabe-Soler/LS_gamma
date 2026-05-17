import pandas as pd
import numpy as np
import yfinance as yf
from arch.univariate import HARX


def fetch_returns(ticker: str, period: str = "2y") -> pd.Series:
    raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"No price data for: {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    prices = raw["Close"].dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna() * 100
    log_returns.name = ticker
    return log_returns


def fit_har(returns: pd.Series, lags: list = [1, 5, 22]):
    model = HARX(returns, lags=lags)
    result = model.fit(disp="off")
    return result


def vol_forecast(ticker: str, horizon: int = 21, period: str = "2y") -> dict:
    returns = fetch_returns(ticker, period=period)
    result = fit_har(returns)

    forecast = result.forecast(horizon=horizon, reindex=False)

    if hasattr(forecast, "variance") and forecast.variance is not None:
        var_path = forecast.variance.iloc[-1].values
        mean_variance = np.mean(var_path)
    else:
        raise ValueError(f"HAR forecast failed for: {ticker}")

    daily_vol = np.sqrt(mean_variance) / 100
    ann_vol = daily_vol * np.sqrt(252)

    raw = yf.download(ticker, period="5d", progress=False, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"No recent price data for: {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    last_price = float(raw["Close"].dropna().iloc[-1])

    return {
        "Ticker": ticker,
        "horizon_days": horizon,
        "har_vol_ann": round(ann_vol, 4),
        "har_vol_daily": round(daily_vol, 6),
        "last_price": round(last_price, 2),
        "aic": round(result.aic, 2),
    }


def forecast_vol_batch(
    tickers: list,
    horizon: int = 21,
    period: str = "2y",
    verbose: bool = True
) -> pd.DataFrame:
    records = []

    for i, ticker in enumerate(tickers):
        try:
            row = vol_forecast(ticker, horizon=horizon, period=period)
            records.append(row)
            if verbose:
                print(f"[{i+1}/{len(tickers)}] {ticker}: HAR vol = {row['har_vol_ann']:.2%}")
        except Exception as e:
            if verbose:
                print(f"[{i+1}/{len(tickers)}] {ticker}: FAILED — {e}")

            records.append({
                "Ticker": ticker,
                "horizon_days": horizon,
                "har_vol_ann": np.nan,
                "har_vol_daily": np.nan,
                "last_price": np.nan,
                "aic": np.nan
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    CSV_PATH = "/Users/gabe/LS_gamma/dev/stock_and_option_data_short_dated.csv"
    HORIZON = 21

    options = pd.read_csv(CSV_PATH)
    tickers = options["Ticker"].tolist()

    print(f"\n=== HAR Volatility Forecasts | horizon={HORIZON}d | {len(tickers)} tickers ===\n")

    har_df = forecast_vol_batch(tickers, horizon=HORIZON, verbose=True)
    har_df.to_csv("har_forecasts.csv", index=False)
    print(har_df.to_string(index=False))

