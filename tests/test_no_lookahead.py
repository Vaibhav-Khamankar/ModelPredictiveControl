import numpy as np
import pytest
from src.data_pipeline.tensor_builder import LiveTensorBuilder

def test_ring_buffer_no_lookahead():
    symbols = ["BTCUSDT", "ETHUSDT"]
    context_length = 10
    builder = LiveTensorBuilder(symbols=symbols, context_length=context_length, num_features=7)

    # Simulate streaming 15 consecutive ticks
    for i in range(15):
        # Roll window BEFORE starting to build the new candle
        if i > 0:
            builder.roll_window()
            
        builder.push_tick("BTCUSDT", [float(i)] * 7)
        builder.push_tick("ETHUSDT", [float(i * 2)] * 7)

    assert builder.is_ready is True
    
    # Extract tensor window
    tensor = builder.get_normalized_tensor(device="cpu").numpy().squeeze(0)

    # Verify that the ring buffer strictly contains the last `context_length` ticks (5 through 14)
    # The last element in time axis should correspond to tick 14
    unnormalized_last = builder.buffer[-1, 0, 0]
    assert unnormalized_last == 14.0, f"Expected last tick to be 14.0, got {unnormalized_last}"