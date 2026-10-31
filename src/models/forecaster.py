import torch
import torch.nn as nn
from src.utils.logger import logger

class AlphaRiskForecaster(nn.Module):
    """
    Consumes the latent representations from the LoRA-adapted Kairos 50M model.
    Projects the sequence into a 10D expected returns vector (Alpha) 
    and a mathematically valid, Positive Semi-Definite 10x10 Covariance Matrix (Risk).
    """
    def __init__(self, base_model: nn.Module, latent_dim: int = 768, num_assets: int = 10):
        super().__init__()
        self.base_model = base_model
        self.num_assets = num_assets
        
        # Alpha Head: Predicts expected returns vector
        self.alpha_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_assets)
        )
        
        # Risk Head: Predicts Cholesky factor elements to guarantee a PSD matrix
        # Elements required for lower triangular matrix: (n * (n + 1)) // 2
        self.cholesky_dim = (num_assets * (num_assets + 1)) // 2
        self.risk_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.cholesky_dim)
        )

    def _make_psd_covariance(self, cholesky_vectors: torch.Tensor) -> torch.Tensor:
        """
        Converts a flat prediction vector into a symmetric, Positive Semi-Definite (PSD) 
        covariance matrix using Cholesky decomposition ($LL^T$).
        """
        batch_size = cholesky_vectors.shape[0]
        L = torch.zeros(batch_size, self.num_assets, self.num_assets, device=cholesky_vectors.device)
        
        # Map output elements to lower triangular indices
        tril_indices = torch.tril_indices(row=self.num_assets, col=self.num_assets, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = cholesky_vectors
        
        # Force diagonal elements to be strictly positive to ensure invertibility
        diag_indices = torch.arange(self.num_assets)
        L[:, diag_indices, diag_indices] = torch.nn.functional.softplus(L[:, diag_indices, diag_indices]) + 1e-5
        
        # L * L^T strictly guarantees a PSD matrix
        covariance = torch.bmm(L, L.transpose(1, 2))
        return covariance

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Expects x shape: [Batch, Sequence_Length, Features]
        Returns:
            alpha: [Batch, Num_Assets] expected returns
            covariance: [Batch, Num_Assets, Num_Assets] PSD risk matrix
        """
        # 1. Extract latent sequence from the adapted Kairos trunk
        latent_state = self.base_model(x)
        
        # 2. Pool the sequence dimension (taking the most recent token representation)
        if latent_state.dim() == 3:
            pooled_state = latent_state[:, -1, :]
        else:
            pooled_state = latent_state
            
        # 3. Project to target financial metrics
        alpha = self.alpha_head(pooled_state)
        cholesky_vecs = self.risk_head(pooled_state)
        
        # 4. Enforce structural mathematics on the covariance prediction
        covariance = self._make_psd_covariance(cholesky_vecs)
        
        return alpha, covariance