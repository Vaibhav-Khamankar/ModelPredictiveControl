import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import glob
import os

def analyze_telemetry(telemetry_file_path: str, symbols: list):
    """Parses live telemetry to audit slippage, deadband rates, and model IC."""
    print(f"Auditing Telemetry: {telemetry_file_path}\n" + "="*50)
    
    total_ticks = 0
    rejected_ticks = 0
    
    buy_slippage = []
    sell_slippage = []
    
    # Store alphas for IC calculation
    timestamps = []
    alpha_matrix = []
    
    with open(telemetry_file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
                
            # 1. Parse Order Executions (Slippage Audit)
            if record.get("event") == "PORTFOLIO_REBALANCE_EXECUTED":
                for order in record.get("executed_orders", []):
                    if order["side"] == "buy":
                        buy_slippage.append(order["slippage_bps"])
                    else:
                        sell_slippage.append(order["slippage_bps"])
                continue
                
            # 2. Parse Tick Inference (Deadband & Alpha Audit)
            if "deadband_metrics" in record:
                total_ticks += 1
                if not record["deadband_metrics"]["rebalance_approved"]:
                    rejected_ticks += 1
                    
                # Collect alpha vector for IC math
                timestamps.append(pd.to_datetime(record["timestamp"]))
                alpha_matrix.append(record["alpha_metrics"]["raw_alpha_vector"])

    # --- METRICS COMPILATION ---
    
    # Deadband Stats
    rejection_rate = (rejected_ticks / total_ticks) * 100 if total_ticks > 0 else 0
    print("--- 1. DEADBAND THRESHOLD ---")
    print(f"Total Evaluated Ticks : {total_ticks}")
    print(f"Trades Rejected       : {rejected_ticks} ({rejection_rate:.2f}%)")
    if rejection_rate < 90:
        print(">> WARNING: Rejection rate below 90%. Optimizer may be over-trading.")
    print("")
    
    # Slippage Stats
    avg_buy = np.mean(buy_slippage) if buy_slippage else 0.0
    avg_sell = np.mean(sell_slippage) if sell_slippage else 0.0
    print("--- 2. EXECUTION SLIPPAGE (bps) ---")
    print(f"Average Buy Slippage  : {avg_buy:+.2f} bps ({len(buy_slippage)} fills)")
    print(f"Average Sell Slippage : {avg_sell:+.2f} bps ({len(sell_slippage)} fills)")
    
    if avg_buy > 12 or avg_sell > 12:
        print(">> FATAL: Realized slippage exceeds 12 bps. Backtest assumptions are broken.")
    print("")

    # --- INFORMATION COEFFICIENT (IC) PREP ---
    print("--- 3. INFORMATION COEFFICIENT (IC) ---")
    # Convert predictions to a structured DataFrame
    df_pred_alpha = pd.DataFrame(alpha_matrix, index=timestamps, columns=symbols)
    
    print(f"Extracted {len(df_pred_alpha)} prediction vectors.")
    print("To calculate live IC, merge this with actual exchange returns:")
    print("  1. Fetch actual 5m close prices for these timestamps.")
    print("  2. Calculate actual forward return: df_actual_returns = df_prices.pct_change().shift(-1)")
    print("  3. Compute Rank IC per row:")
    print("     ic_series = df_pred_alpha.corrwith(df_actual_returns, axis=1, method='spearman')")
    print("     print(f'Mean Live IC: {ic_series.mean():.4f}')")
    
    return df_pred_alpha

if __name__ == "__main__":
    # Auto-find the newest telemetry file
    log_files = glob.glob("logs/telemetry/live_telemetry_*.jsonl")
    if not log_files:
        print("No telemetry files found in logs/telemetry/")
    else:
        latest_log = max(log_files, key=os.path.getctime)
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT", 
                   "DOT/USDT", "DOGE/USDT", "AVAX/USDT", "MATIC/USDT", "LINK/USDT"]
        
        df_alphas = analyze_telemetry(latest_log, symbols)