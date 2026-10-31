import os
import json
import time
import asyncio
import torch
import numpy as np
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from src.utils.logger import logger
from src.data_pipeline.fetchers import BinanceDataFetcher
from src.data_pipeline.tensor_builder import LiveTensorBuilder
from src.optimizer.markowitz_solver import MarkowitzOptimizer

class LiveExecutionEngine:
    """
    Asynchronous trading engine that isolates high-frequency WebSocket ingestion 
    from blocking PyTorch/CVXPY inference tasks using a ThreadPoolExecutor.
    Supports both live Binance Futures execution and local paper trading.
    """
    def __init__(self, model: torch.nn.Module, config: dict, symbols: list):
        self.device = config.get("training", {}).get("device", "mps")
        self.model = model.to(self.device)
        self.model.eval()
        
        self.symbols = symbols
        self.use_testnet = config.get("USE_TESTNET", True)
        self.paper_trading = config.get("PAPER_TRADING", True) # Local paper trading bypass
        
        self.fetcher = BinanceDataFetcher(use_testnet=self.use_testnet)
        self.tensor_builder = LiveTensorBuilder(symbols=self.symbols)
        self.optimizer = MarkowitzOptimizer()
        
        self.compute_pool = ThreadPoolExecutor(max_workers=1)
        self.current_weights = np.zeros(len(self.symbols))
        self.is_trading = False
        
        # Synchronization lock for the 10 distinct websocket streams
        self.closed_candles = set()

        # --- LOCAL PAPER TRADING VIRTUAL ACCOUNT STATE ---
        self.virtual_cash = config.get("trading", {}).get("initial_balance", 10000.0)
        self.virtual_positions = {sym: 0.0 for sym in self.symbols}

        # --- TELEMETRY & AUDIT PATHS ---
        self.telemetry_dir = config.get("telemetry_dir", "logs/telemetry")
        os.makedirs(self.telemetry_dir, exist_ok=True)
        self.telemetry_file = os.path.join(self.telemetry_dir, f"live_telemetry_{int(time.time())}.jsonl")

        # --- LIVE RISK MANAGEMENT STATE ---
        self.transaction_fee = config.get("trading", {}).get("fee_rate", 0.001)
        self.slippage_rate = config.get("trading", {}).get("slippage", 0.001)
        self.min_edge_margin = 0.0005
        
        self.high_water_mark = self.virtual_cash if self.paper_trading else None
        self.max_drawdown_limit = 0.05
        self.cooldown_counter = 0
        self.cooldown_duration = 12

        # --- ADD DYNAMIC MINIMUM TRADE WEIGHT (e.g., 1%) ---
        self.min_trade_weight = config.get("trading", {}).get("min_trade_weight", 0.01)

        # --- PERSISTENT PRICE FALLBACK MAP ---
        self.last_known_prices = {}

        self.alpha_ema = None
        self.ema_smoothing_factor = 0.2  # 0.2 means 20% new prediction, 80% old prediction

    async def start(self):
        # Initialize client if live trading or pre-warming from REST
        await self.fetcher.initialize()
        logger.info(f"Starting execution engine (Paper Trading: {self.paper_trading}). Target assets: {len(self.symbols)}")
        logger.info(f"Telemetry logging to: {self.telemetry_file}")
        
        # --- PRE-WARM TENSOR BUILDER VIA REST API ---
        logger.info("Pre-warming tensor ring buffer with historical REST klines...")
        try:
            for symbol in self.symbols:
                klines_5m = await self.fetcher.client.futures_klines(symbol=symbol.replace('/', ''), interval='5m', limit=96)
                klines_1h = await self.fetcher.client.futures_klines(symbol=symbol.replace('/', ''), interval='1h', limit=24)
                klines_1d = await self.fetcher.client.futures_klines(symbol=symbol.replace('/', ''), interval='1d', limit=14)
                
                formatted_5m = [[float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), 0.0, 0.0] for k in klines_5m]
                formatted_1h = [[float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), 0.0, 0.0] for k in klines_1h]
                formatted_1d = [[float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), 0.0, 0.0] for k in klines_1d]
                
                self.tensor_builder.seed_symbol_history(symbol, formatted_5m, formatted_1h, formatted_1d)
                
            logger.info(f"Tensor builder pre-warmed successfully. Ready state: {self.tensor_builder.is_ready}")
        except Exception as e:
            logger.error(f"Failed to seed historical klines: {e}")
        
        self.is_trading = True
        
        await self.fetcher.start_multiplex_stream(
            symbols=self.symbols, 
            interval="5m", 
            callback=self._handle_websocket_tick
        )

    def _log_telemetry(self, record: dict):
        try:
            with open(self.telemetry_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write telemetry record: {e}")

    def _handle_websocket_tick(self, msg: Dict):
        if "data" not in msg:
            return
            
        stream_type = msg.get("stream", "")
        data = msg["data"]
        
        if "@kline_" in stream_type:
            symbol = data["s"]
            kline = data["k"]
            
            tick_data = [
                float(kline["o"]), float(kline["h"]), float(kline["l"]), 
                float(kline["c"]), float(kline["v"]), 0.0, 0.0 
            ]
            
            # Cache last known close price
            self.last_known_prices[symbol] = tick_data[3]
            self.tensor_builder.push_tick(symbol, tick_data)
            
            if kline.get("x"): 
                self.closed_candles.add(symbol)
                timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                logger.info(f"[{timestamp}] 5m Candle closed for {symbol} ({len(self.closed_candles)}/{len(self.symbols)})")
                
                if len(self.closed_candles) == len(self.symbols):
                    if self.tensor_builder.is_ready:
                        logger.info(f"[{timestamp}] All 10 assets synchronized. Extracting tensor and triggering ONE inference pass.")
                        x_tensor = self.tensor_builder.get_normalized_tensor(device=self.device)
                        asyncio.create_task(self._run_inference_pipeline(x_tensor, timestamp))
                    else:
                        logger.warning(f"[{timestamp}] Tensor builder buffer incomplete. Skipping inference.")
                    
                    self.tensor_builder.roll_window()
                    self.closed_candles.clear()

    async def _get_account_equity(self) -> float:
        """Calculates dynamic Mark-to-Market equity in paper mode with persistent price guards."""
        if self.paper_trading:
            latest_prices = self.tensor_builder.get_latest_prices()
            position_value = 0.0
            
            for sym, qty in self.virtual_positions.items():
                if abs(qty) < 1e-8:
                    continue
                    
                # Prefer live ring buffer price, fallback to persistent tick cache
                price = latest_prices.get(sym) or self.last_known_prices.get(sym, 0.0)
                
                if price <= 0.0:
                    logger.error(f"Unable to resolve price for {sym}! Skipping position valuation guard.")
                    continue
                    
                position_value += qty * price
                
            return float(self.virtual_cash + position_value)
        return await self.fetcher.get_account_equity()

    async def _get_positions(self) -> dict:
        """Returns virtual position state in paper mode or queries exchange."""
        if self.paper_trading:
            return self.virtual_positions.copy()
        return await self.fetcher.get_positions()

    def _blocking_compute_step(self, x_tensor: torch.Tensor, timestamp: str):
        t0 = time.perf_counter()
        
        with torch.no_grad():
            batch_size, seq_len, num_assets, num_features = x_tensor.shape
            x_reshaped = x_tensor.view(batch_size, -1) 
            
            model_outputs = self.model(x_reshaped)
            pred_alpha = model_outputs[0]
            pred_cov = model_outputs[1]
            
        inference_time_ms = (time.perf_counter() - t0) * 1000
        
        raw_alpha_np = pred_alpha.squeeze(0).cpu().numpy()
        cov_np = pred_cov.squeeze(0).cpu().numpy()
        
        # --- EMA SMOOTHING & INITIALIZATION ---
        if not hasattr(self, 'alpha_ema') or self.alpha_ema is None:
            # Lock initial state on first tick without running blending equation
            self.alpha_ema = np.copy(raw_alpha_np)
        else:
            # Apply mathematical smoothing strictly on subsequent ticks
            self.alpha_ema = (self.ema_smoothing_factor * raw_alpha_np) + ((1.0 - self.ema_smoothing_factor) * self.alpha_ema)

        # --- NOISE THRESHOLDING (BACKTEST PARITY) ---
        # Detach a copy from the persistent EMA state for optimization
        alpha_np = np.copy(self.alpha_ema)
        alpha_np[np.abs(alpha_np) < 1e-4] = 0.0
        
        cov_cond_number = float(np.linalg.cond(cov_np))
        cov_diag_min = float(np.min(np.diag(cov_np)))
        cov_diag_max = float(np.max(np.diag(cov_np)))
        
        t_opt_0 = time.perf_counter()
        
        opt_outputs = self.optimizer.optimize(alpha_np, cov_np, self.current_weights)
        theoretical_weights = opt_outputs[0]
        status = opt_outputs[1]
        
        optimizer_time_ms = (time.perf_counter() - t_opt_0) * 1000
        
        current_expected_return = float(np.dot(self.current_weights, alpha_np))
        theoretical_expected_return = float(np.dot(theoretical_weights, alpha_np))
        return_delta = theoretical_expected_return - current_expected_return
        
        total_friction_rate = self.transaction_fee + self.slippage_rate
        proposed_turnover = float(np.sum(np.abs(theoretical_weights - self.current_weights)))
        proposed_costs = proposed_turnover * total_friction_rate
        required_hurdle = proposed_costs + self.min_edge_margin
        
        rebalance_approved = return_delta > required_hurdle
        
        if rebalance_approved:
            target_weights = theoretical_weights
            action_reason = "REBALANCE: Expected return delta exceeded friction hurdle."
        else:
            target_weights = self.current_weights
            action_reason = f"HOLD: Return delta ({return_delta:.6f}) < Hurdle ({required_hurdle:.6f})."
            
        if self.device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        telemetry_payload = {
            "timestamp": timestamp,
            "inference_time_ms": round(inference_time_ms, 2),
            "optimizer_time_ms": round(optimizer_time_ms, 2),
            "solver_status": status,
            "alpha_metrics": {
                "mean": round(float(np.mean(alpha_np)), 6),
                "std": round(float(np.std(alpha_np)), 6),
                "top_asset": self.symbols[int(np.argmax(alpha_np))],
                "top_alpha": round(float(np.max(alpha_np)), 6),
                "raw_model_alpha": [round(x, 6) for x in raw_alpha_np.tolist()],
                "processed_alpha_vector": [round(x, 6) for x in alpha_np.tolist()]
            },
            "covariance_metrics": {
                "condition_number": round(cov_cond_number, 2),
                "diag_min": round(cov_diag_min, 6),
                "diag_max": round(cov_diag_max, 6)
            },
            "deadband_metrics": {
                "current_exp_return": round(current_expected_return, 6),
                "theoretical_exp_return": round(theoretical_expected_return, 6),
                "return_delta": round(return_delta, 6),
                "proposed_turnover": round(proposed_turnover, 6),
                "proposed_costs": round(proposed_costs, 6),
                "required_hurdle": round(required_hurdle, 6),
                "rebalance_approved": rebalance_approved,
                "action_reason": action_reason
            },
            "weights": {
                "current": [round(w, 4) for w in self.current_weights.tolist()],
                "target": [round(w, 4) for w in target_weights.tolist()]
            },
            "risk_metrics": {
                "virtual_cash": round(float(self.virtual_cash), 2)
            },
            "latest_prices": {sym: round(float(price), 4) for sym, price in self.last_known_prices.items()}
        }
        
        return target_weights, status, telemetry_payload

    async def _run_inference_pipeline(self, x_tensor: torch.Tensor, timestamp: str):

        # Short settling window for async multi-stream WebSockets
        await asyncio.sleep(1.0)
        try:
            target_weights, status, telemetry = await asyncio.get_event_loop().run_in_executor(
                                                self.compute_pool, 
                                                self._blocking_compute_step, 
                                                x_tensor,
                                                timestamp
                                            )
            current_equity = await self._get_account_equity()

            logger.info(f"Portfolio State | Cash: ${self.virtual_cash:.2f} | Positions: {self.virtual_positions}")
            
            if self.high_water_mark is None or current_equity > self.high_water_mark:
                self.high_water_mark = current_equity
            
            current_drawdown = (self.high_water_mark - current_equity) / self.high_water_mark if self.high_water_mark > 0 else 0.0
            
            circuit_breaker_active = False
            
            if self.cooldown_counter > 0:
                circuit_breaker_active = True
                target_weights = np.zeros(len(self.symbols))
                self.cooldown_counter -= 1
                logger.warning(f"[{timestamp}] CIRCUIT BREAKER COOLDOWN: {self.cooldown_counter} ticks remaining.")
            elif current_drawdown > self.max_drawdown_limit:
                circuit_breaker_active = True
                logger.error(f"[{timestamp}] CIRCUIT BREAKER TRIGGERED! Drawdown: {current_drawdown*100:.2f}%. Liquidating.")
                target_weights = np.zeros(len(self.symbols))
                self.cooldown_counter = self.cooldown_duration
                self.high_water_mark = current_equity 
                
            telemetry["risk_metrics"] = {
                "current_equity": round(current_equity, 4),
                "high_water_mark": round(self.high_water_mark, 4),
                "current_drawdown_pct": round(current_drawdown * 100, 2),
                "circuit_breaker_active": circuit_breaker_active,
                "cooldown_counter": self.cooldown_counter
            }
            
            self._log_telemetry(telemetry)
            
            logger.info(
                f"[{timestamp}] Equity: ${current_equity:.2f} | DD: {current_drawdown*100:.2f}% | "
                f"Top Alpha: {telemetry['alpha_metrics']['top_asset']} | "
                f"Action: {telemetry['deadband_metrics']['action_reason']}"
            )
            
            if not np.allclose(self.current_weights, target_weights, atol=1e-4):
                await self._execute_portfolio_rebalance(target_weights, current_equity)
                
        except Exception as e:
            logger.error(f"Inference pipeline failed: {e}", exc_info=True)

    async def _execute_portfolio_rebalance(self, target_weights: np.ndarray, current_equity: float):
        logger.info(f"Executing rebalance. Target weights: {target_weights}")
        try:
            current_positions = await self._get_positions()
            latest_prices = self.tensor_builder.get_latest_prices()
            
            execution_audit = []
            
            for i, symbol in enumerate(self.symbols):
                target_weight = target_weights[i]
                
                # FIX 1: Apply fallback price map so orders are never skipped on transient tick drops
                current_price = latest_prices.get(symbol) or self.last_known_prices.get(symbol, 0.0)
                
                if current_price <= 0.0:
                    logger.error(f"Cannot execute order for {symbol}: price unresolvable.")
                    continue
                    
                target_notional = current_equity * target_weight
                target_size = target_notional / current_price
                current_size = current_positions.get(symbol, 0.0)
                
                delta_size = target_size - current_size
                delta_notional = abs(delta_size * current_price)
                
                min_notional_required = current_equity * self.min_trade_weight
                if delta_notional < min_notional_required:
                    continue
                    
                side = "buy" if delta_size > 0 else "sell"
                t_order_start = time.time()
                
                if self.paper_trading:
                    slippage_factor = (1 + self.slippage_rate) if side == "buy" else (1 - self.slippage_rate)
                    realized_fill_price = current_price * slippage_factor
                    fee_cost = delta_notional * self.transaction_fee
                    
                    cash_delta = -(delta_size * realized_fill_price) - fee_cost
                    self.virtual_cash += cash_delta
                    self.virtual_positions[symbol] = target_size
                    
                    execution_record = {
                        "symbol": symbol,
                        "side": side,
                        "mode": "PAPER_TRADE",
                        "expected_price": current_price,
                        "filled_price": round(realized_fill_price, 4),
                        "slippage_bps": round(self.slippage_rate * 10000, 2),
                        "fee_cost_usd": round(fee_cost, 4),
                        "order_size_tokens": round(abs(delta_size), 4),
                        "order_notional_usd": round(delta_notional, 2),
                        "latency_ms": round((time.time() - t_order_start) * 1000, 2)
                    }
                    # With this explicit notional breakdown:
                    logger.info(
                        f"[PAPER FILL] {side.upper()} {abs(delta_size):.4f} {symbol} "
                        f"@ ${realized_fill_price:.2f} | Notional: ${delta_notional:.2f}"
                    )
                else:
                    order_res = await self.fetcher.client.futures_create_order(
                        symbol=symbol.replace('/', ''), 
                        side=side.upper(), 
                        type='MARKET',
                        quantity=round(abs(delta_size), 3)
                    )
                    realized_fill_price = float(order_res.get("avgPrice", current_price))
                    execution_record = {
                        "symbol": symbol,
                        "side": side,
                        "mode": "LIVE_EXCHANGE",
                        "expected_price": current_price,
                        "filled_price": realized_fill_price,
                        "order_size_tokens": abs(delta_size),
                        "order_notional_usd": round(delta_notional, 2)
                    }
                    
                execution_audit.append(execution_record)
                
            # FIX 2: Derive current weights from ACTUAL position values to prevent phantom state sync
            total_eval_equity = await self._get_account_equity()
            if total_eval_equity > 0:
                actual_weights = []
                for sym in self.symbols:
                    p = latest_prices.get(sym) or self.last_known_prices.get(sym, 0.0)
                    pos_val = self.virtual_positions.get(sym, 0.0) * p
                    actual_weights.append(pos_val / total_eval_equity)
                self.current_weights = np.array(actual_weights)

            logger.info("Portfolio rebalance completed successfully.")
            
            self._log_telemetry({
                "event": "PORTFOLIO_REBALANCE_EXECUTED",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "executed_orders": execution_audit
            })
            
        except Exception as e:
            logger.error(f"Critical error during order execution: {e}")

    async def shutdown(self):
        self.is_trading = False
        await self.fetcher.close()
        self.compute_pool.shutdown(wait=True)
        logger.info("Live trader shut down securely.")