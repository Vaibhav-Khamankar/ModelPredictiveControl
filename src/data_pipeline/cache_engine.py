import os
import torch
from src.utils.logger import logger

def generate_sliding_windows():
    logger.info("Loading processed raw tensors...")
    X_raw = torch.load("data/processed/X_raw.pt")     # [Steps, Assets, Features] -> [52690, 10, 7]
    Y_alpha = torch.load("data/processed/Y_alpha.pt") # [Steps, Assets]
    Y_cov = torch.load("data/processed/Y_cov.pt")     # [Steps, Assets, Assets]
    
    num_steps, num_assets, num_features = X_raw.shape
    context_len = 96
    
    # Flatten the asset and feature dimensions for the sequence input
    # New shape: [52690, 70]
    X_flat = X_raw.view(num_steps, num_assets * num_features)
    
    num_windows = num_steps - context_len + 1
    logger.info(f"Unrolling {num_windows} sliding windows of length {context_len}...")
    
    # Pre-allocate the memory for the overlapping windows (~1.4 GB footprint)
    X_windows = torch.zeros((num_windows, context_len, num_assets * num_features), dtype=torch.float32)
    
    for i in range(num_windows):
        X_windows[i] = X_flat[i : i + context_len]
        
    # Align target labels to the final step of each window (index 95, 96, 97...)
    Y_alpha_aligned = Y_alpha[context_len - 1 : ]
    Y_cov_aligned = Y_cov[context_len - 1 : ]
    
    logger.info(f"Final X shape: {X_windows.shape}")
    logger.info(f"Final Y_alpha shape: {Y_alpha_aligned.shape}")
    logger.info(f"Final Y_cov shape: {Y_cov_aligned.shape}")
    
    os.makedirs("data/embeddings", exist_ok=True)
    
    # Saving to the embeddings folder (technically sequenced inputs, not latent embeddings)
    torch.save(X_windows, "data/embeddings/X_windows.pt")
    torch.save(Y_alpha_aligned, "data/embeddings/Y_alpha_aligned.pt")
    torch.save(Y_cov_aligned, "data/embeddings/Y_cov_aligned.pt")
    
    logger.info("Sliding windows successfully cached. Phase 1 trainer is ready.")

if __name__ == "__main__":
    generate_sliding_windows()