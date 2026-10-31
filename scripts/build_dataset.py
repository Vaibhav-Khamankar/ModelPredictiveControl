import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils.logger import logger

def build_aligned_dataset():
    symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", 
        "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT"
    ]
    
    logger.info("1. Loading and aligning CSVs...")
    dataframes = {}
    
    for sym in symbols:
        df = pd.read_csv(f"data/raw/{sym}_5m.csv", index_col="timestamp", parse_dates=True)
        dataframes[sym] = df
        
    # Strictly intersect indices to drop any mismatched rows (like the extra BNB candle)
    common_index = dataframes[symbols[0]].index
    for sym in symbols[1:]:
        common_index = common_index.intersection(dataframes[sym].index)
        
    logger.info(f"Common index established. Total aligned steps: {len(common_index)}")
    
    # 2. Build 3D Feature Tensor: [Steps, Assets, Features]
    num_steps = len(common_index)
    num_assets = len(symbols)
    num_features = 7 # [O, H, L, C, V, OI, Funding]
    
    X_raw = np.zeros((num_steps, num_assets, num_features), dtype=np.float32)
    close_prices = np.zeros((num_steps, num_assets), dtype=np.float32)
    
    for asset_idx, sym in enumerate(symbols):
        aligned_df = dataframes[sym].loc[common_index]
        X_raw[:, asset_idx, :] = aligned_df.values
        close_prices[:, asset_idx] = aligned_df['close'].values

    logger.info("2. Calculating target labels (Alpha & Covariance)...")
    
    # Calculate returns: (Close_t - Close_{t-1}) / Close_{t-1}
    # Shifted by -1 so that row T predicts T+1
    returns = np.zeros_like(close_prices)
    returns[:-1] = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
    returns[-1] = 0.0 # Drop last row later since we don't have T+1
    
    # Y_Alpha: The forward T+1 expected return
    Y_alpha = returns
    
    # Y_Cov: The realized rolling covariance matrix (Lookahead is fine here, these are targets)
    # We use a 12-period (1 hour) forward window to calculate the realized risk
    forward_window = 12
    Y_cov = np.zeros((num_steps, num_assets, num_assets), dtype=np.float32)
    
    for i in tqdm(range(num_steps - forward_window), desc="Computing Realized Covariance"):
        window_returns = returns[i : i + forward_window]
        # Calculate covariance of returns over the forward window
        cov_matrix = np.cov(window_returns, rowvar=False)
        # Ensure numerical stability for PSD
        cov_matrix = cov_matrix + np.eye(num_assets) * 1e-6
        Y_cov[i] = cov_matrix

    # 3. Truncate invalid tail rows and save
    valid_steps = num_steps - forward_window
    X_raw = X_raw[:valid_steps]
    Y_alpha = Y_alpha[:valid_steps]
    Y_cov = Y_cov[:valid_steps]
    
    logger.info(f"3. Saving processed tensors to data/processed/. Final shape: {X_raw.shape}")
    
    os.makedirs("data/processed", exist_ok=True)
    torch.save(torch.tensor(X_raw), "data/processed/X_raw.pt")
    torch.save(torch.tensor(Y_alpha), "data/processed/Y_alpha.pt")
    torch.save(torch.tensor(Y_cov), "data/processed/Y_cov.pt")

if __name__ == "__main__":
    build_aligned_dataset()