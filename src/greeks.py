import numpy as np
import pandas as pd 
from scipy.stats import norm


# Core Black-Scholes math
def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_delta(S: float, K: float, T: float, r: float,
             sigma: float, option_type: str) -> float:

    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = _d1(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:

    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_theta(S: float, K: float, T: float, r: float,
             sigma: float, option_type: str) -> float:
  
    if T <= 0 or sigma <= 0:
        return np.nan
    d1  = _d1(S, K, T, r, sigma)
    d2  = _d2(S, K, T, r, sigma)
    pdf = norm.pdf(d1)

    if option_type == "call":
        theta_annual = (
            -(S * pdf * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )
    else:
        theta_annual = (
            -(S * pdf * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )

    return theta_annual / 365   # daily theta


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
   
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100   # per 1% IV move

