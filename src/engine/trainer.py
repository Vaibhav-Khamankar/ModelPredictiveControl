import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm
from src.utils.logger import logger
from src.engine.loss import DifferentiableSharpeLoss, DifferentiableSortinoLoss, CalibratedAsymmetricLoss

class PhaseOneTrainer:
    """
    Supervised learning loop for the AlphaRiskForecaster.
    Trains strictly the LoRA adapters via backpropagation using pre-computed Kairos embeddings.
    """
    def __init__(self, model: nn.Module, config: dict, device: str = "mps"):
        self.model = model.to(device)
        self.device = device
        
        # Only pass parameters that require gradients (the LoRA adapters) to the optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        lr = config.get("training", {}).get("learning_rate", 0.0003)
        weight_decay = config.get("training", {}).get("weight_decay", 0.01)
        self.grad_clip = config.get("training", {}).get("grad_clip_norm", 1.0)
        
        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        
        # Differentiable Calibrated Asymmetric Loss (105120 5-min periods/year)
        self.alpha_criterion = CalibratedAsymmetricLoss(periods_per_year=105120, l2_reg=0.005)
        
    def _covariance_loss(self, pred_cov: torch.Tensor, target_cov: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Frobenius norm between the predicted PSD covariance matrix 
        and the realized target covariance matrix.
        """
        diff = pred_cov - target_cov
        return torch.linalg.matrix_norm(diff, ord='fro').mean()

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_alpha_loss = 0.0
        total_cov_loss = 0.0

        progress_bar = tqdm(train_loader, desc="Training", leave=False, dynamic_ncols=True)

        for batch in progress_bar:
            # Unpack and move to MPS
            x_emb, y_alpha, y_cov = [b.to(self.device) for b in batch]

            self.optimizer.zero_grad()
            pred_alpha, pred_cov = self.model(x_emb)
            
            # Composite Loss Calculation
            loss_alpha = self.alpha_criterion(pred_alpha, y_alpha)
            loss_cov = self._covariance_loss(pred_cov, y_cov)
            
            # Weighted combination: Optimize negative Sharpe + Frobenius Risk norm
            total_loss = loss_alpha + (0.1 * loss_cov)
            
            # Backpropagation
            total_loss.backward()
            
            # Gradient Clipping prevents exploding gradients from volatility spikes
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()

            # Update progress bar (loss_alpha is negative Sortino ratio)
            progress_bar.set_postfix({
            "NegSortino": f"{loss_alpha.item():.4f}",
            "Risk": f"{loss_cov.item():.4f}"
        })
            
            total_alpha_loss += loss_alpha.item()
            total_cov_loss += loss_cov.item()

        # Clean cache ONCE per epoch after backprop completes
        if self.device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        avg_alpha_loss = total_alpha_loss / len(train_loader)
        avg_cov_loss = total_cov_loss / len(train_loader)

        return avg_alpha_loss, avg_cov_loss

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluates the model on the validation set without computing gradients.
        """
        self.model.eval()
        total_alpha_loss = 0.0
        total_cov_loss = 0.0
        
        with torch.no_grad():
            for x_emb, y_alpha, y_cov in val_loader:
                x_emb = x_emb.to(self.device)
                y_alpha = y_alpha.to(self.device)
                y_cov = y_cov.to(self.device)
                
                pred_alpha, pred_cov = self.model(x_emb)
                
                loss_alpha = self.alpha_criterion(pred_alpha, y_alpha)
                loss_cov = self._covariance_loss(pred_cov, y_cov)
                
                total_alpha_loss += loss_alpha.item()
                total_cov_loss += loss_cov.item()
                
        return total_alpha_loss / len(val_loader), total_cov_loss / len(val_loader)