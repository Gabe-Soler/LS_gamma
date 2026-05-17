import pandas as pd
import yfinance as yf
import numpy as np 
from datetime import datetime, timedelta
from pathlib import Path

#need to import wrds, option metrics, bloomber ect later on, 


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
 
 
# SPY intraday prices  (1-min bars, last 5 trading days)
# Used for: hourly delta re-hedging, intraday realised vol calculation
 
def get_spy_intraday(interval: str = "1m", period: str = "5d") -> pd.DataFrame:

    print(f"  Fetching SPY intraday ({interval} bars, {period})...")
    df = yf.download("SPY", period=period, interval=interval,
                     progress=False, auto_adjust=True)
    df.index.name = "datetime"
    df.to_parquet(DATA_DIR / "spy_intraday.parquet")
    df.to_csv(DATA_DIR / "spy_intraday.csv")
    print(f"    → {len(df)} rows  |  {df.index[0]} → {df.index[-1]}")
    return df
 
 
# SPY daily prices  (3-year history)
# vol forecast models (GARCH, HAR, etc.)
 
def get_spy_daily(years: int = 3) -> pd.DataFrame:
    start = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    print(f"  Fetching SPY daily prices ({years}yr from {start})...")
    df = yf.download("SPY", start=start, interval="1d",
                     progress=False, auto_adjust=True)
    df.index.name = "date"
    df["returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df.to_parquet(DATA_DIR / "spy_daily.parquet")
    df.to_csv(DATA_DIR / "spy_daily.csv")
    print(f"    → {len(df)} trading days")
    return df
 
 
# sPY options chain
#   IV , ATM straddle, OTM strangle entry, greeks
#    need to calculate greeks by inversing black-scholes, (mk mid prc)
 
def get_options_chain(
    target_dte_min: int = 7,
    target_dte_max: int = 28,
) -> dict[str, pd.DataFrame]:

    print(f"  Fetching SPY options chain ({target_dte_min}-{target_dte_max} DTE)...")
    spy = yf.Ticker("SPY")
    today  = datetime.today().date()
    chains = {}
 
    for expiry in spy.options:
        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        dte      = (exp_date - today).days
 
        if not (target_dte_min <= dte <= target_dte_max):
            continue

        chain = spy.option_chain(expiry)
 
        calls       = chain.calls.copy()
        puts        = chain.puts.copy()
        calls["type"] = "call"
        puts["type"]  = "put"
        combined    = pd.concat([calls, puts], ignore_index=True)
 
        
        cols = ["type", "strike", "bid", "ask",
                "impliedVolatility", "volume", "openInterest"]
        combined = combined[cols].rename(columns={
            "impliedVolatility": "iv",
            "openInterest":      "oi",
        })
        combined["mid"] = (combined["bid"] + combined["ask"]) / 2
        combined["expiry"] = expiry
        combined["dte"] = dte
 
        chains[expiry] = combined
        print(f"    → {expiry}  (DTE={dte})  {len(combined)} contracts")
 
    # Save each expiry as its own parquet
    for expiry, df in chains.items():
        df.to_parquet(DATA_DIR / f"options_{expiry}.parquet")
        df.to_csv(DATA_DIR / f"options_{expiry}.csv")

    print(f"    → {len(chains)} expiries fetched")
    return chains
 
 
def get_atm_strike(spy_price: float, chain: pd.DataFrame) -> float:
    strikes    = chain["strike"].unique()
    atm_strike = strikes[np.argmin(np.abs(strikes - spy_price))]
    return atm_strike
 
 
def get_otm_strikes(spy_price: float, chain: pd.DataFrame,
                    pct_otm: float = 0.05) -> tuple[float, float]:

    strikes      = chain["strike"].unique()
    call_target  = spy_price * (1 + pct_otm)
    put_target   = spy_price * (1 - pct_otm)
    call_strike  = strikes[np.argmin(np.abs(strikes - call_target))]
    put_strike   = strikes[np.argmin(np.abs(strikes - put_target))]
    return call_strike, put_strike
 
 
# VIX daily
#  secondary IV reference, regime signal
 
def get_vix(years: int = 3) -> pd.DataFrame:

    start = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    print(f"  Fetching VIX ({years}yr)...")
    df = yf.download("^VIX", start=start, interval="1d",
                     progress=False, auto_adjust=True)[["Close"]]
    df.columns = ["vix_close"]
    df.index.name = "date"
    df.to_parquet(DATA_DIR / "vix_daily.parquet")
    df.to_csv(DATA_DIR / "vix_daily.csv")
    print(f"    → {len(df)} rows")
    return df
 
 
# R-F rate 
#  Black-Scholes IV calculation, strategy PnL
 
def get_risk_free_rate(years: int = 3) -> pd.DataFrame:
    
    start = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    print(f"  Fetching risk-free rate (^IRX, {years}yr)...")
    df = yf.download("^IRX", start=start, interval="1d",
                     progress=False, auto_adjust=True)[["Close"]]
    df.columns  = ["rate_pct"]
    df["rate_decimal"] = df["rate_pct"] / 100
    df.index.name  = "date"
    df.to_parquet(DATA_DIR / "risk_free_rate.parquet")
    df.to_csv(DATA_DIR / "risk_free_rate.csv")
    print(f"    → {len(df)} rows  |  latest: {df['rate_pct'].iloc[-1]:.2f}%")
    return df
 
 
# SPY dividend calendar
#  Used for: accurate options pricing (dividends affect put/call parity)
 
def get_dividends() -> pd.DataFrame:

    print("  Fetching SPY dividends...")
    spy = yf.Ticker("SPY")
    df  = spy.dividends.to_frame().rename(columns={"Dividends": "dividend"})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df.to_parquet(DATA_DIR / "spy_dividends.parquet")
    df.to_csv(DATA_DIR / "spy_dividends.csv")
    print(f"    → {len(df)} dividend records")
    return df
 
 
#load all saved data from disk
 
def load_all() -> dict:
  
    data = {}
 
    # Scalar datasets
    for key, fname in [
        ("spy_intraday",   "spy_intraday.parquet"),
        ("spy_daily",      "spy_daily.parquet"),
        ("vix",            "vix_daily.parquet"),
        ("risk_free_rate", "risk_free_rate.parquet"),
        ("dividends",      "spy_dividends.parquet"),
    ]:
        path = DATA_DIR / fname
        if path.exists():
            data[key] = pd.read_parquet(path)
        else:
            print(f"  Warning: {fname} not found — run fetch_all() first")

    options_files = sorted(DATA_DIR.glob("options_*.parquet"))

    data["options"] = {
        f.stem.replace("options_", ""): pd.read_parquet(f)
        for f in options_files
    }
    if data["options"]:
        print(f"  Loaded {len(data['options'])} options expiries from disk")
 
    return data
 
 
# only run at start of day
def fetch_all() -> dict:

    print("\nFetching all strategy data...")
    data = {
        "spy_intraday":   get_spy_intraday(),
        "spy_daily":      get_spy_daily(),
        "options":        get_options_chain(),
        "vix":            get_vix(),
        "risk_free_rate": get_risk_free_rate(),
        "dividends":      get_dividends(),
    }
    print("\nAll data fetched and saved to data\n")
    return data
 
 
#refreshes data
if __name__ == "__main__":
    fetch_all()
 