import yaml
import cvxpy as cp
import numpy as np
from typing import List, Dict
from src.utils.logger import logger

class ConstraintBuilder:
    """
    Constructs mathematical boundaries and penalty terms for the convex portfolio optimizer 
    based on the configuration parameters, guaranteeing mathematically strict bounds.
    """
    def __init__(self, config_path: str = "configs/optimization_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.bounds = config.get("bounds", {})
        self.cash = config.get("cash_management", {})
        self.penalties = config.get("penalties", {})
        
        self._validate_feasibility()
        
    def _validate_feasibility(self):
        """
        Validates that the combination of configured limits is mathematically solvable.
        """
        min_w = self.bounds.get("min_weight_per_asset", 0.0)
        max_w = self.bounds.get("max_weight_per_asset", 0.25)
        max_exposure = self.bounds.get("max_gross_exposure", 1.0)
        reserve_cash = self.cash.get("reserve_cash_pct", 0.10)
        
        investable = max_exposure - reserve_cash
        
        if min_w * 10 > investable:
            raise ValueError(f"Infeasible: 10 assets * {min_w} minimum weight > {investable} investable capital.")
        if max_w * 10 < investable:
            raise ValueError(f"Infeasible: 10 assets * {max_w} maximum weight < {investable} investable capital.")
            
    def build_constraints(self, w: cp.Variable) -> List[cp.Constraint]:
        """
        Builds the hard mathematical constraints for the CVXPY optimization problem.
        """
        constraints = []
        
        max_exposure = self.bounds.get("max_gross_exposure", 1.0)
        reserve_cash = self.cash.get("reserve_cash_pct", 0.10)
        investable_capital = max_exposure - reserve_cash
        
        # 1. Total allocation constraint (Cash reserve + Asset weights = 1.0)
        constraints.append(cp.sum(w) == investable_capital)
        
        # 2. Shorting and lower bounds
        allow_shorting = self.bounds.get("allow_shorting", False)
        min_weight = self.bounds.get("min_weight_per_asset", 0.0)
        
        if not allow_shorting:
            constraints.append(w >= min_weight)
            
        # 3. Maximum concentration limit (Crucial for crypto)
        max_weight = self.bounds.get("max_weight_per_asset", 0.25)
        constraints.append(w <= max_weight)
        
        logger.debug(f"Constraints built: Exposure={investable_capital}, Bounds=[{min_weight}, {max_weight}]")
        return constraints
        
    def build_penalty_terms(self, w_current: cp.Variable, w_previous: np.ndarray) -> cp.Expression:
        """
        Constructs the convex penalty expression for transaction costs and turnover.
        This expression is subtracted from the objective function.
        """
        if w_previous is None:
            return 0.0
            
        # Convert BPS to decimal multiplier
        tx_cost = self.penalties.get("transaction_cost_bps", 10) / 10000.0
        slippage = self.penalties.get("slippage_bps", 5) / 10000.0
        turnover_gamma = self.penalties.get("turnover_penalty", 0.05)
        
        execution_penalty = tx_cost + slippage
        
        # L1 Norm (Absolute difference) between new target weights and previous weights
        turnover = cp.norm(w_current - w_previous, 1)
        
        # Penalty term to subtract from expected returns
        total_penalty = (execution_penalty * turnover) + (turnover_gamma * cp.sum_squares(w_current - w_previous))
        
        return total_penalty