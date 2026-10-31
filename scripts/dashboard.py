import streamlit as st
import pandas as pd
import json
import glob
import os
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="Live Bot Monitor", layout="wide")
st.title("🟢 Live Execution Dashboard")

auto_refresh = st.sidebar.checkbox("Auto-Refresh (10s)", value=True)

def load_latest_telemetry():
    log_files = glob.glob("logs/telemetry/live_telemetry_*.jsonl")
    if not log_files:
        return None, [], []
    
    latest_file = max(log_files, key=os.path.getctime)
    ticks, trades = [], []
    
    with open(latest_file, "r") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                if record.get("event") == "PORTFOLIO_REBALANCE_EXECUTED":
                    trades.append(record)
                elif "deadband_metrics" in record:
                    ticks.append(record)
            except:
                continue
    return latest_file, ticks, trades

file_path, ticks, trades = load_latest_telemetry()

if not ticks:
    st.warning("Waiting for the bot to write its first 5-minute telemetry tick...")
else:
    st.sidebar.success(f"Reading: {os.path.basename(file_path)}")
    latest_tick = ticks[-1]
    
    # --- 1. TOP METRICS ROW ---
    col1, col2, col3, col4 = st.columns(4)
    
    current_equity = latest_tick.get("risk_metrics", {}).get("current_equity", 0.0)
    drawdown = latest_tick.get("risk_metrics", {}).get("current_drawdown_pct", 0.0)
    
    rejections = sum(1 for t in ticks if not t["deadband_metrics"]["rebalance_approved"])
    rejection_rate = (rejections / len(ticks)) * 100 if ticks else 0
    
    # Replace line 50:
    all_slippage = [order.get("slippage_bps", 0.0) for tr in trades for order in tr.get("executed_orders", [])]
    avg_slippage = sum(all_slippage) / len(all_slippage) if all_slippage else 0.0

    col1.metric("Live Equity", f"${current_equity:,.2f}")
    col2.metric("Current Drawdown", f"{drawdown:.2f}%")
    col3.metric("Deadband Rejection Rate", f"{rejection_rate:.1f}%")
    col4.metric("Avg Realized Slippage", f"{avg_slippage:.2f} bps")

    st.divider()
    
    symbols = ["BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "DOGE", "AVAX", "BNB", "LINK"]
    
    alpha_metrics = latest_tick.get("alpha_metrics", {})
    # Fall back sequentially across new schema, old schema, and zeroed defaults
    alphas = (
        alpha_metrics.get("processed_alpha_vector") or 
        alpha_metrics.get("raw_alpha_vector") or 
        alpha_metrics.get("raw_model_alpha") or 
        [0.0] * len(symbols)
    )
    
    weights_data = latest_tick.get("weights", {})
    current_weights = weights_data.get("current", [0.0] * len(symbols))
    target_weights = weights_data.get("target", [0.0] * len(symbols))
    
    # --- CURRENT PORTFOLIO INVENTORY & LIQUIDITY ---
    st.subheader("Current Portfolio Inventory")
    
    risk_metrics = latest_tick.get("risk_metrics", {})
    cash_val = float(risk_metrics.get("virtual_cash", current_equity * (1.0 - sum(current_weights))))
    positions_val = max(0.0, current_equity - cash_val)
    cash_pct = (cash_val / current_equity * 100) if current_equity > 0 else 100.0
    crypto_pct = (positions_val / current_equity * 100) if current_equity > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Portfolio Equity", f"${current_equity:.2f}")
    c2.metric("Available Cash (USDT)", f"${cash_val:.2f}", f"{cash_pct:.1f}% cash drag")
    c3.metric("Crypto Exposure", f"${positions_val:.2f}", f"{crypto_pct:.1f}% allocated")
    c4.metric("Active Holdings", f"{sum(1 for w in current_weights if w > 0.001)} / {len(symbols)}")

    # 1. Isolate the Cash Row into its own un-sorted DataFrame
    df_cash = pd.DataFrame([{
        "Asset": "USDT (Cash)",
        "Token Inventory": "—",
        "Current Price": "$1.0000",
        "Mark Value": cash_val, 
        "Portfolio Weight": f"{cash_pct:.2f}%"
    }])

    tick_prices = latest_tick.get("latest_prices", {})
    
    # 2. Collect only the active Crypto Assets
    crypto_rows = []
    for i, sym in enumerate(symbols):
        if current_weights[i] > 0.0001:
            pos_usd = current_weights[i] * current_equity
            price = float(tick_prices.get(f"{sym}USDT", 0.0))
            qty = (pos_usd / price) if price > 0 else 0.0
            
            crypto_rows.append({
                "Asset": sym,
                "Token Inventory": f"{qty:.4f}" if qty > 0 else "N/A",
                "Current Price": f"${price:.4f}" if price > 0 else "N/A",
                "Mark Value": pos_usd,
                "Portfolio Weight": f"{current_weights[i]*100:.2f}%"
            })

    # 3. Sort Crypto Assets mathematically, then force Cash to the top
    if crypto_rows:
        df_crypto = pd.DataFrame(crypto_rows).sort_values(by="Mark Value", ascending=False)
        df_inventory = pd.concat([df_cash, df_crypto], ignore_index=True)
    else:
        df_inventory = df_cash
        
    # Format strings last to preserve numeric sorting integrity
    df_inventory["Mark Value"] = df_inventory["Mark Value"].map("${:.2f}".format)
    
    st.dataframe(df_inventory, width="stretch", hide_index=True)
    st.divider()

    # --- 2. LIVE PORTFOLIO HOLDINGS ---
    st.subheader("Live Portfolio Allocation")
    
    portfolio_data = []
    for i, sym in enumerate(symbols):
        weight_pct = current_weights[i] * 100
        notional_usd = current_weights[i] * current_equity
        target_pct = target_weights[i] * 100
        
        # Keep Allocated Capital as a raw float
        portfolio_data.append({
            "Asset": sym,
            "Current Weight": f"{weight_pct:.2f}%",
            "Target Weight": f"{target_pct:.2f}%",
            "Allocated Capital": notional_usd,
            "Latest Predicted Alpha": f"{alphas[i]:.6f}"
        })
            
    df_portfolio = pd.DataFrame(portfolio_data)
    # Sort mathematically first
    df_portfolio = df_portfolio.sort_values(by="Allocated Capital", ascending=False)
    # Format strings second
    df_portfolio["Allocated Capital"] = df_portfolio["Allocated Capital"].map("${:.2f}".format)
    
    st.dataframe(df_portfolio, width="stretch", hide_index=True)
    st.divider()

    # --- 3. LIVE ALPHA PREDICTIONS CHART ---
    st.subheader("Latest Alpha Predictions (Live Edge)")
    st.caption(f"Last updated: {latest_tick['timestamp']}")
    
    df_alphas = pd.DataFrame({"Asset": symbols, "Predicted Alpha": alphas})
    df_alphas.set_index("Asset", inplace=True)
    st.bar_chart(df_alphas)

    st.divider()

    # --- 4. EXECUTION DECISION LOG ---
    st.subheader("Latest Deadband Decision")
    db = latest_tick["deadband_metrics"]
    st.json({
        "Action": db["action_reason"],
        "Expected Return Delta": f"{db['return_delta']:.6f}",
        "Friction Hurdle": f"{db['required_hurdle']:.6f}"
    })
        
    st.divider()

    # --- 5. ORDER HISTORY (SCROLLABLE WITH PINNED TOTALS) ---
    st.subheader("History")
    if trades:
        recent_orders = []
        total_buys_usd = 0.0
        total_sells_usd = 0.0

        # Increased to pull the last 50 rebalances to populate the scroll box
        # Increased to pull the last 50 rebalances to populate the scroll box
        for tr in reversed(trades[-50:]):
            raw_timestamp = tr.get("timestamp", "N/A")
            
            # Parse UTC string and convert to IST (+5:30)
            formatted_time = raw_timestamp
            if raw_timestamp != "N/A":
                try:
                    dt_utc = datetime.strptime(raw_timestamp, "%Y-%m-%dT%H:%M:%SZ")
                    dt_ist = dt_utc + timedelta(hours=5, minutes=30)
                    formatted_time = dt_ist.strftime("%Y-%m-%d %H:%M:%S IST")
                except ValueError:
                    pass # Fallback to raw string if format unexpectedly changes

            for order in tr.get("executed_orders", []):
                side = str(order.get("side", "")).upper()
                qty = float(order.get("order_size_tokens", 0.0))
                price = float(order.get("filled_price", 0.0))
                notional = float(order.get("order_notional_usd", qty * price))

                if side == "BUY":
                    total_buys_usd += notional
                elif side == "SELL":
                    total_sells_usd += notional

                recent_orders.append({
                    "Time": formatted_time,
                    "Symbol": order.get("symbol", "N/A"),
                    "Side": side,
                    "Quantity": f"{qty:.4f}",
                    "Filled Price": f"${price:.4f}",
                    "Total Value": f"${notional:.2f}"
                })

        if recent_orders:
            df_orders = pd.DataFrame(recent_orders)
            net_spent = total_buys_usd - total_sells_usd

            def color_side(val):
                if val == 'BUY':
                    return 'color: #ff4b4b;' # Red
                elif val == 'SELL':
                    return 'color: #00ff00;' # Green
                return ''

            styled_df = df_orders.style.map(color_side, subset=['Side'])

            # height=400 restricts the box to ~10 rows and forces a scrollbar
            st.dataframe(styled_df, hide_index=True, width="stretch", height=400)
            
            # Pinned totals permanently visible below the scroll box
            st.markdown(
                f"**TOTAL REBALANCE OUTLAY:** "
                f":red[Buys: ${total_buys_usd:.2f}] | "
                f":green[Sells: ${total_sells_usd:.2f}] | "
                f"**Net: ${net_spent:.2f}**"
            )
        else:
            st.info("No orders executed in recent rebalances.")
    else:
        st.info("No trades executed yet.")

if auto_refresh:
    time.sleep(10)
    st.rerun()