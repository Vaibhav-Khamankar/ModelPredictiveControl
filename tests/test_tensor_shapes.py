import torch
import pytest
from src.models.forecaster import AlphaRiskForecaster

class MockKairosBase(torch.nn.Module):
    def __init__(self, latent_dim=768):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        return torch.randn(batch_size, seq_len, self.latent_dim)

def test_alpha_and_covariance_shapes():
    batch_size = 4
    context_len = 96
    num_assets = 10
    num_features = 7
    latent_dim = 768

    # Synthetic input tensor: [Batch, Sequence, Assets, Features] -> flattened sequence
    dummy_input = torch.randn(batch_size, context_len, num_assets * num_features)
    
    mock_base = MockKairosBase(latent_dim=latent_dim)
    forecaster = AlphaRiskForecaster(base_model=mock_base, latent_dim=latent_dim, num_assets=num_assets)
    
    alpha, cov = forecaster(dummy_input)

    # 1. Assert Alpha (Expected Returns) dimensions: [Batch, Num_Assets]
    assert alpha.shape == (batch_size, num_assets), f"Expected Alpha shape {(batch_size, num_assets)}, got {alpha.shape}"

    # 2. Assert Covariance Matrix dimensions: [Batch, Num_Assets, Num_Assets]
    assert cov.shape == (batch_size, num_assets, num_assets), f"Expected Covariance shape {(batch_size, num_assets, num_assets)}, got {cov.shape}"

def test_covariance_psd_property():
    batch_size = 2
    num_assets = 10
    mock_base = MockKairosBase()
    forecaster = AlphaRiskForecaster(base_model=mock_base, latent_dim=768, num_assets=num_assets)
    
    dummy_input = torch.randn(batch_size, 96, 70)
    _, cov = forecaster(dummy_input)

    for b in range(batch_size):
        mat = cov[b]
        # Check symmetry: A == A^T
        assert torch.allclose(mat, mat.T, atol=1e-5), "Predicted covariance matrix is not symmetric!"
        
        # Check positive semi-definiteness via eigenvalues >= 0
        eigenvalues = torch.linalg.eigvalsh(mat)
        assert torch.all(eigenvalues >= 0), f"Covariance matrix contains negative eigenvalues: {eigenvalues}"