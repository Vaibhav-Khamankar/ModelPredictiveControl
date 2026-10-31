import torch
from tqdm import tqdm
from src.utils.logger import logger

class BacktestEngine:
    def __init__(self, model, config, device="mps"):
        self.model = model
        self.config = config
        self.device = device
        # 10 bps standard crypto taker fee
        self.transaction_fee = config.get("trading", {}).get("fee_rate", 0.001)
        # 10 bps assumed spread crossing and market impact
        self.slippage_rate = 0.001

    def optimize_portfolio(self, alpha: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """
        Calculates optimal asset weights using unconstrained Mean-Variance Optimization.
        alpha shape: [10], cov shape: [10, 10]
        """
        # 1. Enforce strict mathematical symmetry (A = A^T)
        cov_sym = 0.5 * (cov + cov.T)
        
        # 2. Apply a dynamic Tikhonov ridge penalty rather than a static 1e-5.
        # This scales the penalty to the actual matrix values, guaranteeing a 
        # strictly positive diagonal that the LU decomposition can always invert.
        dynamic_penalty = torch.abs(torch.diag(cov_sym)).max() * 0.05 + 1e-3
        cov_reg = cov_sym + torch.eye(cov_sym.size(0), device=self.device) * dynamic_penalty
        
        try:
            # Precision matrix is the inverse of covariance
            precision_matrix = torch.linalg.inv(cov_reg)
            
            # w = Cov^-1 * Expected_Returns
            raw_weights = torch.mv(precision_matrix, alpha)
            
            # SIGNAL THRESHOLDING: Zero out microscopic noise predictions
            # If the model is predicting < 0.01% return, do not risk capital on it.
            raw_weights[torch.abs(alpha) < 1e-4] = 0.0
            
            # Normalize to 100% gross exposure (Long-Short Market Neutral)
            total_exposure = torch.sum(torch.abs(raw_weights))
            if total_exposure > 0:
                weights = raw_weights / total_exposure
            else:
                weights = torch.zeros_like(raw_weights)
                
            return weights
            
        except RuntimeError as e:
            # If the model output is still completely degenerate, hold cash.
            logger.warning(f"Covariance inversion failed despite dynamic regularization: {e}")
            return torch.zeros_like(alpha)

    def run(self, test_loader):
        self.model.eval()
        net_portfolio_value = 1.0    
        gross_portfolio_value = 1.0  
        benchmark_value = 1.0        # Beta tracker
        cumulative_fees = 0.0
        
        equity_curve = [net_portfolio_value]
        benchmark_curve = [benchmark_value]
        current_weights = torch.zeros(10, device=self.device)
        
        # Static 10% allocation across all 10 assets
        benchmark_weights = torch.ones(10, device=self.device) / 10.0
        
        # --- CIRCUIT BREAKER STATE ---
        high_water_mark = 1.0
        cooldown_counter = 0
        max_drawdown_limit = 0.05  # Liquidate if we drop 5% from the peak
        cooldown_duration = 12     # Lockout for 1 hour (12 x 5-minute candles)
        
        logger.info(f"Starting chronological simulation. Initial Capital: 1.0000x")
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Simulating Ticks", dynamic_ncols=True)
            
            for batch in progress_bar:
                x_emb, y_alpha, y_cov = [b.to(self.device) for b in batch]
                
                # 1. Forecast next step
                pred_alpha, pred_cov = self.model(x_emb)
                
                # The L2 penalty already calibrated the logits. Pass them raw.
                calibration_scalar = 1.0 
                alpha = pred_alpha.squeeze(0) * calibration_scalar
                
                cov = pred_cov.squeeze(0)
                actual_return = y_alpha.squeeze(0)
                
                # 2. Portfolio Optimization (Theoretical Optimal Allocation)
                theoretical_weights = self.optimize_portfolio(alpha, cov)
                
                # 3. DEADBAND THRESHOLD
                current_expected_return = torch.dot(current_weights, alpha)
                theoretical_expected_return = torch.dot(theoretical_weights, alpha)
                
                total_friction_rate = self.transaction_fee + self.slippage_rate
                proposed_turnover = torch.sum(torch.abs(theoretical_weights - current_weights))
                proposed_costs = proposed_turnover * total_friction_rate
                
                min_edge_margin = 0.0005  
                
                if (theoretical_expected_return - current_expected_return) > (proposed_costs + min_edge_margin):
                    target_weights = theoretical_weights
                else:
                    target_weights = current_weights
                
                # --- CIRCUIT BREAKER OVERRIDE STAYS HERE ---
                
                # 4. Calculate Realized Transaction Costs + Slippage
                turnover = torch.sum(torch.abs(target_weights - current_weights))
                costs = turnover * total_friction_rate
                cumulative_fees += costs.item() # Now represents total friction
                
                # 5. Calculate Step Return & Equity
                portfolio_return = torch.dot(target_weights, actual_return)
                gross_portfolio_value = gross_portfolio_value * (1 + portfolio_return.item())
                
                step_pnl = portfolio_return - costs
                net_portfolio_value = net_portfolio_value * (1 + step_pnl.item())
                
                # Update Benchmark
                benchmark_return = torch.dot(benchmark_weights, actual_return)
                benchmark_value = benchmark_value * (1 + benchmark_return.item())
                
                # --- UPDATE PEAK EQUITY STAYS HERE ---
                
                equity_curve.append(net_portfolio_value)
                benchmark_curve.append(benchmark_value)
                current_weights = target_weights
                
                progress_bar.set_postfix({
                    "Net": f"{net_portfolio_value:.4f}",
                    "B&H": f"{benchmark_value:.4f}"
                })

        # Calculate final performance metrics
        net_return_pct = (net_portfolio_value - 1.0) * 100
        gross_return_pct = (gross_portfolio_value - 1.0) * 100
        benchmark_pct = (benchmark_value - 1.0) * 100
        alpha_spread = net_portfolio_value - benchmark_value
        
        logger.info("=== BACKTEST METRICS DEEP DIVE ===")
        logger.info(f"Buy & Hold Benchmark  : {benchmark_value:.4f}x ({benchmark_pct:.2f}%)")
        logger.info(f"Gross Portfolio Value : {gross_portfolio_value:.4f}x ({gross_return_pct:.2f}%)")
        logger.info(f"Total Friction (Fees+): {cumulative_fees:.4f}")
        logger.info(f"Net Portfolio Value   : {net_portfolio_value:.4f}x ({net_return_pct:.2f}%)")
        logger.info(f"Strategy Alpha Spread : {alpha_spread:+.4f}x over market")
        
        return equity_curve