import numpy as np
import pytest
from src.optimizer.markowitz_solver import MarkowitzOptimizer

def test_markowitz_solver_bounds():
    optimizer = MarkowitzOptimizer("configs/optimization_config.yaml")
    
    num_assets = 10
    np.random.seed(42)
    
    # Synthetic inputs
    alpha = np.random.uniform(-0.02, 0.05, size=num_assets)
    random_mat = np.random.randn(num_assets, num_assets)
    covariance = np.dot(random_mat, random_mat.T) # Force PSD
    
    weights, status = optimizer.optimize(alpha, covariance)

    assert status in ["optimal", "optimal_inaccurate"]
    
    # 1. Verify weights sum to investable capital (1.0 - cash reserve)
    investable_capital = 1.0 - optimizer.constraint_builder.cash.get("reserve_cash_pct", 0.10)
    assert pytest.approx(np.sum(weights), abs=1e-4) == investable_capital
    
    # 2. Verify maximum single asset weight bound (0.25 max limit)
    max_weight_limit = optimizer.constraint_builder.bounds.get("max_weight_per_asset", 0.25)
    assert np.all(weights <= max_weight_limit + 1e-5), f"Max weight exceeded: {np.max(weights)}"
    
    # 3. Verify no shorting rule (all weights >= 0)
    assert np.all(weights >= -1e-5), f"Negative weight detected: {np.min(weights)}"