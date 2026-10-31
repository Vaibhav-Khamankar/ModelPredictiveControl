import yaml
import numpy as np
import cvxpy as cp
from typing import Tuple, Optional
from src.optimizer.constraints import ConstraintBuilder
from src.utils.logger import logger

class MarkowitzOptimizer:
    """
    Phase 2 Portfolio Optimizer: Solves the convex quadratic program to determine
    the optimal capital allocation across 10 cryptocurrency assets.
    """
    def __init__(self, config_path: str = "configs/optimization_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.constraint_builder = ConstraintBuilder(config_path)
        self.risk_aversion = self.config.get("risk_management", {}).get("target_volatility", 0.40)
        
    def optimize(
        self, 
        alpha: np.ndarray, 
        covariance: np.ndarray, 
        w_previous: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Solves the Markowitz Mean-Variance objective:
            Maximize: w^T * alpha - (risk_aversion / 2) * w^T * Sigma * w - Penalty(w, w_prev)
            Subject to: Configured constraints (Exposure limits, Asset bounds, Cash reserve)

        Args:
            alpha: [10,] Expected returns vector from Phase 1 Forecaster.
            covariance: [10, 10] Positive Semi-Definite risk matrix from Phase 1.
            w_previous: [10,] Previous portfolio weight allocation (for turnover penalty).

        Returns:
            Tuple of (optimal_weights_array, solver_status)
        """
        num_assets = len(alpha)
        
        # 1. Define Optimization Variable
        w = cp.Variable(num_assets)
        
        # 2. Build Constraints
        constraints = self.constraint_builder.build_constraints(w)
        
        # 3. Calculate Portfolio Variance Term
        # Ensure covariance is symmetric and PSD for CVXPY atom compliance
        cov_psd = cp.psd_wrap(covariance)
        portfolio_variance = cp.quad_form(w, cov_psd)
        
        # 4. Calculate Penalty Terms (Transaction Costs & Turnover)
        penalty_term = self.constraint_builder.build_penalty_terms(w, w_previous)
        
        # 5. Formulate Objective Function
        expected_return = w @ alpha
        objective = cp.Maximize(expected_return - (self.risk_aversion * 0.5 * portfolio_variance) - penalty_term)
        
        # 6. Solve Problem
        problem = cp.Problem(objective, constraints)
        
        try:
            # Try solving with CLARABEL solver (default for modern CVXPY versions)
            problem.solve(solver=cp.CLARABEL, verbose=False)
        except Exception as e:
            logger.warning(f"CLARABEL solver failed: {e}. Retrying with OSQP...")
            try:
                problem.solve(solver=cp.OSQP, verbose=False)
            except Exception as e2:
                logger.error(f"OSQP solver failed as well: {e2}")

        # 7. Extract Solution & Implement Fallback Execution
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            optimal_weights = w.value
            # Clean up small floating point anomalies WITHOUT destroying negative weights
            optimal_weights[np.abs(optimal_weights) < 1e-8] = 0.0
            
            # DO NOT clip to 0.0. If you need a max gross exposure limit, do it via cvxpy constraints.
            
            logger.info(f"Convex optimization successful. Status: {problem.status}")
            return optimal_weights, problem.status
        
        else:
            logger.error(f"Optimization failed! Solver Status: {problem.status}. Activating Safety Fallback...")
            
            # Fallback Logic: Maintain previous portfolio weights or return equal allocation
            if w_previous is not None:
                logger.warning("Fallback: Maintaining previous portfolio weights.")
                return w_previous, "FALLBACK_PREVIOUS_WEIGHTS"
            else:
                logger.warning("Fallback: Distributing allocation equally across assets.")
                equal_weight = np.full(num_assets, (1.0 - self.constraint_builder.cash.get("reserve_cash_pct", 0.10)) / num_assets)
                return equal_weight, "FALLBACK_EQUAL_WEIGHTS"