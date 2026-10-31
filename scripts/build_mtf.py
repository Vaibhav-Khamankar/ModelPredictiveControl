import pandas as pd
import torch
import os
import glob
from tqdm import tqdm

def main():
    print("1. Discovering and merging all asset CSVs...")
    
    # Grab all 5m CSVs in the raw directory
    csv_files = glob.glob("data/raw/*_5m.csv")
    
    if not csv_files:
        print("Error: No CSV files found matching 'data/raw/*_5m.csv'")
        return

    df_list = []
    agg_dict = {}
    
    for file in csv_files:
        # Extract the symbol from the filename (e.g., ADAUSDT)
        symbol = os.path.basename(file).split('_')[0]
        df = pd.read_csv(file, index_col="timestamp", parse_dates=True)
        
        # Rename generic columns to include the asset symbol
        df.columns = [f"{symbol}_{col}" for col in df.columns]
        df_list.append(df)
        
        # Dynamically map the correct aggregation method for the macro resampling
        agg_dict[f"{symbol}_open"] = 'first'
        agg_dict[f"{symbol}_high"] = 'max'
        agg_dict[f"{symbol}_low"] = 'min'
        agg_dict[f"{symbol}_close"] = 'last'
        agg_dict[f"{symbol}_volume"] = 'sum'
        # Add the two features injected by the download script
        agg_dict[f"{symbol}_open_interest"] = 'last'
        agg_dict[f"{symbol}_funding_rate"] = 'last'

    print(f"Found {len(csv_files)} assets. Concatenating horizontally...")
    
    # Merge all assets side-by-side on the datetime index
    df_5m = pd.concat(df_list, axis=1)
    
    # Forward-fill and backward-fill gaps if some coins have missing 5m candles
    df_5m = df_5m.ffill().bfill()
    
    print(f"Merged dataset shape: {df_5m.shape}")

    print("2. Resampling to Macro Timeframes (1H, 1D)...")
    # Shift(1) is CRITICAL to prevent look-ahead bias (leaking the current unclosed candle)
    df_1h = df_5m.resample('1h', closed='right', label='right').agg(agg_dict).ffill()
    df_1d = df_5m.resample('1d', closed='right', label='right').agg(agg_dict).ffill()
    
    print("3. Fusing Multi-Timeframe Tensors and Target Labels...")
    X_fused, Y_alpha_list, Y_cov_list, Y_step_list = [], [], [], []

    # Isolate the 'close' columns for the 10 assets to calculate target returns
    close_cols = [c for c in df_5m.columns if c.endswith('_close')]

    start_idx = 4032 

    # We subtract 12 at the end to ensure we always have 1 hour of future data for the target
    for i in tqdm(range(start_idx, len(df_5m) - 12), desc="Building MTF Sequence"):
        current_time = df_5m.index[i]
        
        window_5m = df_5m.iloc[i - 95 : i + 1].values
        window_1h = df_1h.loc[:current_time].tail(24).values
        window_1d = df_1d.loc[:current_time].tail(14).values
        
        if len(window_1h) == 24 and len(window_1d) == 14 and len(window_5m) == 96:
            # --- 1. BUILD X INPUT ---
            t_1d = torch.tensor(window_1d, dtype=torch.float32)
            t_1h = torch.tensor(window_1h, dtype=torch.float32)
            t_5m = torch.tensor(window_5m, dtype=torch.float32)
            fused_sequence = torch.cat([t_1d, t_1h, t_5m], dim=0)
            
            # --- 2. BUILD Y TARGETS (Next 1 Hour) ---
            # Slice the current candle + next 12 forward candles
            period_closes = df_5m[close_cols].iloc[i : i+13]
            
            # Alpha Target: Smoothed cumulative return over the hour
            cum_returns = (period_closes.iloc[-1] - period_closes.iloc[0]) / period_closes.iloc[0]
            
            # Risk Target: Realized covariance matrix of the forward 5-minute steps
            step_returns = period_closes.pct_change().dropna()
            cov_matrix = step_returns.cov()
            
            # Ensure no corrupted data (halts in trading, missing prints)
            if not cum_returns.isna().any() and not cov_matrix.isna().any().any():
                
                # ACTUAL 5-MINUTE RETURN FOR BACKTEST SIMULATION
                step_5m_return = (period_closes.iloc[1] - period_closes.iloc[0]) / period_closes.iloc[0]
                
                X_fused.append(fused_sequence)
                Y_alpha_list.append(torch.tensor(cum_returns.values, dtype=torch.float32))
                Y_cov_list.append(torch.tensor(cov_matrix.values, dtype=torch.float32))
                Y_step_list.append(torch.tensor(step_5m_return.values, dtype=torch.float32))
                
    X_tensor = torch.stack(X_fused)
    Y_alpha_tensor = torch.stack(Y_alpha_list)
    Y_cov_tensor = torch.stack(Y_cov_list)
    Y_step_tensor = torch.stack(Y_step_list)

    print(f"4. Saving Synchronized Tensors: {X_tensor.shape[0]} sequences")
    os.makedirs("data/embeddings", exist_ok=True)
    torch.save(X_tensor, "data/embeddings/X_windows_mtf_raw.pt")
    torch.save(Y_alpha_tensor, "data/embeddings/Y_alpha_mtf.pt")
    torch.save(Y_cov_tensor, "data/embeddings/Y_cov_mtf.pt")
    torch.save(Y_step_tensor, "data/embeddings/Y_step_mtf.pt")
    print("Done. Ready for Phase 1.")

if __name__ == "__main__":
    main()