import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple
from collections import deque
from src.utils.logger import logger

class LiveTensorBuilder:
    """
    Highly optimized ring-buffer for constructing the [1, 96, 10, 7] PyTorch tensor
    during live WebSocket streaming without Pandas reallocation overhead.
    """
    def __init__(self, symbols: List[str], context_length: int = 96, num_features: int = 7):
        self.symbols = sorted([s.upper() for s in symbols])
        self.context_length = context_length
        self.num_features = num_features
        self.num_assets = len(self.symbols)
        
        self.symbol_to_idx = {sym: i for i, sym in enumerate(self.symbols)}
        
        # Ring buffer: Shape [96, 10, 7]
        self.buffer = np.zeros((self.context_length, self.num_assets, self.num_features), dtype=np.float32)
        
        # Track how many historical candles have been seeded per asset
        self.seeded_counts = {sym: 0 for sym in self.symbols}

    @property
    def is_ready(self) -> bool:
        """
        Returns True only when all assets have exactly context_length (96) candles seeded.
        """
        return all(count >= self.context_length for count in self.seeded_counts.values())

    def seed_symbol_history(self, symbol: str, history_5m: list, history_1h: list, history_1d: list):
        """
        Pre-populates the single MTF buffer array with historical 5m REST klines on startup.
        (Currently ingests only 5m data to match the [96, 10, 7] network architecture).
        """
        if symbol not in self.symbol_to_idx:
            return
            
        idx = self.symbol_to_idx[symbol]
        
        # Ensure we don't try to insert more than 96 candles
        valid_history = history_5m[-self.context_length:]
        num_candles = len(valid_history)
        
        if num_candles == 0:
            return
            
        # Insert historical data at the correct offset
        start_idx = self.context_length - num_candles
        self.buffer[start_idx:, idx, :] = np.array(valid_history, dtype=np.float32)
        
        self.seeded_counts[symbol] = num_candles

    def push_tick(self, symbol: str, tick_data: List[float]) -> None:
        """
        Updates the current open candle with real-time multiplex data.
        tick_data format: [open, high, low, close, volume, open_interest, funding_rate]
        """
        if symbol not in self.symbol_to_idx:
            return
            
        idx = self.symbol_to_idx[symbol]
        # Update the most recent candle (index -1)
        self.buffer[-1, idx, :] = tick_data

    def roll_window(self) -> None:
        """
        Called when a new candle closes. Shifts the buffer back by 1 and clears the latest row.
        """
        self.buffer = np.roll(self.buffer, shift=-1, axis=0)
        self.buffer[-1, :, :] = 0.0 # Clear new slot for the incoming open candle
        
        # Ensure readiness isn't falsely triggered before seeding is complete
        for sym in self.symbols:
            if self.seeded_counts[sym] < self.context_length:
                self.seeded_counts[sym] += 1

    def get_latest_prices(self) -> Dict[str, float]:
        """
        Returns the most recent Close price for all assets, used by the execution engine to size orders.
        Close price is feature index 3.
        """
        latest_prices = {}
        for sym, idx in self.symbol_to_idx.items():
            latest_prices[sym] = float(self.buffer[-1, idx, 3])
        return latest_prices

    def get_normalized_tensor(self, device: str = "mps") -> torch.Tensor:
        """
        Applies per-asset, per-feature Min-Max scaling to the current window
        and returns the PyTorch tensor ready for the MPS forward pass.
        """
        if not self.is_ready:
            raise ValueError("Buffer not yet filled with minimum context length.")

        window = self.buffer.copy()
        
        # Calculate min/max along the time dimension (axis=0) keeping dimensions for broadcasting
        min_v = np.min(window, axis=0, keepdims=True)
        max_v = np.max(window, axis=0, keepdims=True)
        
        # Avoid division by zero
        range_v = np.where((max_v - min_v) == 0, 1e-8, max_v - min_v)
        normalized = (window - min_v) / range_v
        
        # Add batch dimension -> [1, 96, 10, 7]
        normalized = np.expand_dims(normalized, axis=0)
        return torch.tensor(normalized, dtype=torch.float32, device=device)


def process_historical_batch(raw_dataframes: Dict[str, pd.DataFrame], context_length: int = 96) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Offline builder for processing historical DataFrame dumps into overlapping sliding windows.
    Returns: X_tensor, Y_tensor (for Phase 1 pre-caching and LoRA training)
    """
    symbols = sorted(list(raw_dataframes.keys()))
    num_assets = len(symbols)
    
    # Ensure all dataframes are aligned on the same timestamp index
    aligned_df = pd.concat(raw_dataframes, axis=1, join='inner')
    num_samples = len(aligned_df)
    
    if num_samples < context_length + 1:
        raise ValueError("Insufficient historical data to form a single window.")

    logger.info(f"Building historical tensors from {num_samples} aligned candles for {num_assets} assets...")

    # We expect 7 columns per asset in the dataframe
    num_features = 7 
    raw_matrix = np.zeros((num_samples, num_assets, num_features), dtype=np.float32)

    for a_idx, sym in enumerate(symbols):
        raw_matrix[:, a_idx, :] = aligned_df[sym].values

    # Construct sliding windows
    X_batches = []
    Y_batches = []

    for i in range(num_samples - context_length):
        window = raw_matrix[i : i + context_length, :, :]
        target = raw_matrix[i + context_length, :, 3] # Target: Next candle's Close price for all assets
        
        # Normalize window
        min_v = np.min(window, axis=0, keepdims=True)
        max_v = np.max(window, axis=0, keepdims=True)
        range_v = np.where((max_v - min_v) == 0, 1e-8, max_v - min_v)
        norm_window = (window - min_v) / range_v
        
        X_batches.append(norm_window)
        
        # Targets are converted to percentage returns relative to the last close in the window
        last_close = window[-1, :, 3]
        target_return = (target - last_close) / np.where(last_close == 0, 1e-8, last_close)
        Y_batches.append(target_return)

    return torch.tensor(np.array(X_batches), dtype=torch.float32), torch.tensor(np.array(Y_batches), dtype=torch.float32)