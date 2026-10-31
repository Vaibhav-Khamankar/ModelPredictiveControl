import numpy as np
import pandas as pd
from typing import Dict, Union

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.05, periods_per_year: int = 105120) -> float:
    """
    Calculates the annualized Sharpe Ratio for time-series portfolio returns.
    Default periods_per_year (105,120) corresponds to 5-minute candle intervals (365 days).
    """
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period
    std_dev = np.std(returns, ddof=1)
    
    if std_dev == 0:
        return 0.0
        
    sharpe = (np.mean(excess_returns) / std_dev) * np.sqrt(periods_per_year)
    return float(sharpe)

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.05, periods_per_year: int = 105120) -> float:
    """
    Calculates the annualized Sortino Ratio, penalizing only downside volatility.
    """
    if len(returns) < 2:
        return 0.0
        
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_per_period
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return 0.0
        
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    if downside_std == 0:
        return 0.0
        
    sortino = (np.mean(excess_returns) / downside_std) * np.sqrt(periods_per_year)
    return float(sortino)

def calculate_max_drawdown(portfolio_values: np.ndarray) -> Dict[str, float]:
    """
    Calculates Maximum Drawdown (MDD) percentage and peak-to-trough drop.
    """
    if len(portfolio_values) < 2:
        return {"max_drawdown_pct": 0.0, "peak_value": 0.0, "trough_value": 0.0}
        
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    max_dd_idx = np.argmin(drawdowns)
    
    max_drawdown_pct = abs(float(drawdowns[max_dd_idx]))
    trough_value = float(portfolio_values[max_dd_idx])
    peak_value = float(running_max[max_dd_idx])
    
    return {
        "max_drawdown_pct": max_drawdown_pct,
        "peak_value": peak_value,
        "trough_value": trough_value
    }

def calculate_turnover(weights_history: np.ndarray) -> float:
    """
    Calculates average portfolio turnover per rebalancing period across all N assets.
    weights_history shape: [Time_Steps, Num_Assets]
    """
    if len(weights_history) < 2:
        return 0.0
        
    weight_changes = np.abs(np.diff(weights_history, axis=0))
    # Sum across assets for total turn per step, then average across time
    turnover_per_step = np.sum(weight_changes, axis=1)
    return float(np.mean(turnover_per_step))

def evaluate_portfolio_performance(
    portfolio_values: np.ndarray,
    weights_history: np.ndarray,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 105120
) -> Dict[str, float]:
    """
    Executes a comprehensive quantitative valuation report across the strategy run.
    """
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    
    mdd_metrics = calculate_max_drawdown(portfolio_values)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    turnover = calculate_turnover(weights_history)
    
    calmar_ratio = (total_return / mdd_metrics["max_drawdown_pct"]) if mdd_metrics["max_drawdown_pct"] > 0 else 0.0
    
    return {
        "total_return_pct": float(total_return * 100),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar_ratio,
        "max_drawdown_pct": mdd_metrics["max_drawdown_pct"] * 100,
        "average_turnover_per_step": turnover
    }