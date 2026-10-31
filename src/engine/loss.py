import torch
import torch.nn as nn

class DifferentiableSharpeLoss(nn.Module):
    def __init__(self, periods_per_year=105120, risk_free_rate=0.0):
        super().__init__()
        # 105,120 periods for 5-minute crypto candles (24 * 12 * 365)
        self.annualization_factor = torch.sqrt(torch.tensor(periods_per_year, dtype=torch.float32))
        self.risk_free_rate = risk_free_rate

    def forward(self, pred_alpha, actual_returns):
        """
        pred_alpha: [Batch, Assets] - Raw model output
        actual_returns: [Batch, Assets] - y_alpha target tensor
        """
        # 1. Differentiable Position Sizing 
        # Bound predictions between -1 (100% Short) and 1 (100% Long)
        weights = torch.tanh(pred_alpha)
        
        # Normalize so total absolute leverage doesn't exceed 1.0 (100% Capital)
        exposure = torch.sum(torch.abs(weights), dim=1, keepdim=True) + 1e-6
        normalized_weights = weights / exposure
        
        # 2. Calculate Realized Portfolio Returns across the batch
        # Shape: [Batch, Assets] * [Batch, Assets] -> sum over assets -> [Batch]
        portfolio_returns = torch.sum(normalized_weights * actual_returns, dim=1)
        
        # 3. Calculate Mean and Volatility over the batch sequence
        mean_return = torch.mean(portfolio_returns)
        # Add epsilon to prevent NaN gradients if the model outputs zeros
        volatility = torch.std(portfolio_returns) + 1e-6
        
        # 4. Calculate Annualized Sharpe Ratio
        sharpe_ratio = (mean_return - self.risk_free_rate) / volatility
        annualized_sharpe = sharpe_ratio * self.annualization_factor.to(pred_alpha.device)
        
        # 5. Return NEGATIVE Sharpe (PyTorch optimizers minimize loss)
        return -annualized_sharpe

class DifferentiableSortinoLoss(nn.Module):
    def __init__(self, periods_per_year=105120, risk_free_rate=0.0):
        super().__init__()
        self.annualization_factor = torch.sqrt(torch.tensor(periods_per_year, dtype=torch.float32))
        self.risk_free_rate = risk_free_rate

    def forward(self, pred_alpha, actual_returns):
        weights = torch.tanh(pred_alpha)
        
        # Normalize exposure
        exposure = torch.sum(torch.abs(weights), dim=1, keepdim=True) + 1e-6
        normalized_weights = weights / exposure
        
        # Calculate Realized Portfolio Returns
        portfolio_returns = torch.sum(normalized_weights * actual_returns, dim=1)
        mean_return = torch.mean(portfolio_returns)
        
        # SORTINO DIFFERENCE: Isolate strictly negative returns for the volatility penalty
        downside_returns = torch.minimum(portfolio_returns, torch.zeros_like(portfolio_returns))
        
        # Calculate Downside Deviation
        downside_deviation = torch.sqrt(torch.mean(downside_returns**2)) + 1e-6
        
        # Calculate and Annualize Sortino
        sortino_ratio = (mean_return - self.risk_free_rate) / downside_deviation
        annualized_sortino = sortino_ratio * self.annualization_factor.to(pred_alpha.device)
        
        return -annualized_sortino

class CalibratedAsymmetricLoss(nn.Module):
    def __init__(self, periods_per_year=105120, risk_free_rate=0.0, l2_reg=0.01):
        super().__init__()
        self.annualization_factor = torch.sqrt(torch.tensor(periods_per_year, dtype=torch.float32))
        self.risk_free_rate = risk_free_rate
        self.l2_reg = l2_reg  # Penalizes saturated extreme predictions

    def forward(self, pred_alpha, actual_returns):
        # 1. Soft-bounded allocations (allows dynamic scaling)
        weights = torch.tanh(pred_alpha)
        
        # Exposure normalization
        exposure = torch.sum(torch.abs(weights), dim=1, keepdim=True) + 1e-6
        normalized_weights = weights / exposure
        
        # Realized Portfolio Step Returns
        portfolio_returns = torch.sum(normalized_weights * actual_returns, dim=1)
        mean_return = torch.mean(portfolio_returns)
        
        # 2. Isolate Downside Risks
        downside_returns = torch.minimum(portfolio_returns, torch.zeros_like(portfolio_returns))
        portfolio_dev = torch.sqrt(torch.mean(downside_returns**2))
        
        # DYNAMIC ANCHOR: Calculate the raw market's downside deviation across all 10 assets
        market_downside = torch.minimum(actual_returns, torch.zeros_like(actual_returns))
        market_dev = torch.sqrt(torch.mean(market_downside**2))
        
        # Prevent the denominator from collapsing to 1e-6 by flooring it at 10% of actual market risk
        downside_deviation = portfolio_dev + (market_dev * 0.10) + 1e-6
        
        # 3. Base Sortino Calculation
        sortino_ratio = (mean_return - self.risk_free_rate) / downside_deviation
        annualized_sortino = sortino_ratio * self.annualization_factor.to(pred_alpha.device)
        
        # 4. CALIBRATION PENALTY: Prevent logits from saturating at +1/-1
        logit_penalty = torch.mean(pred_alpha ** 2) * self.l2_reg
        
        # Minimize Negative Annualized Sortino + Logit Regularization
        return -annualized_sortino + logit_penalty