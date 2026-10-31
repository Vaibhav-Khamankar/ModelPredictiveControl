import os
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import sys
import yaml
import torch
import asyncio
import argparse
from dotenv import load_dotenv
from torch.utils.data import TensorDataset, DataLoader

# ==============================================================================
# 1. ENVIRONMENT INITIALIZATION & LOGGING (IMPORT LOGGER FIRST)
# ==============================================================================
load_dotenv(override=True)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from src.utils.logger import logger
from src.models.kairos_base import KairosBaseWrapper
from src.models.lora_layers import inject_lora_adapters
from src.models.forecaster import AlphaRiskForecaster 
from src.engine.trainer import PhaseOneTrainer
from src.engine.backtest_loop import BacktestEngine
from src.engine.live_trader import LiveExecutionEngine

def load_base_model() -> AlphaRiskForecaster:
    """
    Constructs the end-to-end model pipeline: Base -> LoRA -> Forecaster.
    """
    logger.info("Initializing Kairos Model Pipeline...")
    base_wrapper = KairosBaseWrapper(device="mps")
    adapted_wrapper = inject_lora_adapters(base_wrapper, "configs/kairos_lora_config.yaml")
    
    forecaster = AlphaRiskForecaster(base_model=adapted_wrapper, latent_dim=128, num_assets=10)
    return forecaster

def run_training():
    logger.info("=== STARTING PHASE 1: LORA ADAPTER TRAINING ===")
    
    with open("configs/kairos_lora_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    logger.info("Loading cached tensor windows into memory...")
    X_windows = torch.load("data/embeddings/X_windows_mtf_raw.pt")
    Y_alpha = torch.load("data/embeddings/Y_alpha_mtf.pt")
    Y_cov = torch.load("data/embeddings/Y_cov_mtf.pt")

    split_idx = int(len(X_windows) * 0.8)
    
    train_dataset = TensorDataset(X_windows[:split_idx], Y_alpha[:split_idx], Y_cov[:split_idx])
    val_dataset = TensorDataset(X_windows[split_idx:], Y_alpha[split_idx:], Y_cov[split_idx:])

    batch_size = config.get("training", {}).get("batch_size", 64)
    epochs = config.get("training", {}).get("epochs", 3)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = load_base_model()
    trainer = PhaseOneTrainer(model, config, device="mps")

    logger.info(f"Starting LoRA training loop for {epochs} epochs on MPS (Batch Size: {batch_size})...")
    
    best_val_loss = float('inf')
    best_path = "data/models/kairos_lora_weights_best"
    final_path = "data/models/kairos_lora_weights_final"
    
    for epoch in range(1, epochs + 1):
        train_alpha_loss, train_cov_loss = trainer.train_epoch(train_loader)
        val_alpha_loss, val_cov_loss = trainer.evaluate(val_loader)
        
        combined_val_loss = abs(val_alpha_loss + val_cov_loss)
        
        logger.info(f"Epoch {epoch}/{epochs} | "
                    f"Train [NegSortino: {train_alpha_loss:.6f}, Risk: {train_cov_loss:.6f}] | "
                    f"Val [NegSortino: {val_alpha_loss:.6f}, Risk: {val_cov_loss:.6f}]")

        if combined_val_loss < best_val_loss:
            best_val_loss = combined_val_loss
            logger.info(f"New best composite validation loss: {best_val_loss:.6f}. Saving best weights...")
            os.makedirs(best_path, exist_ok=True)
            model.base_model.base_model.save_pretrained(best_path)

    logger.info("Phase 1 Training complete. Saving final PEFT LoRA adapter weights...")
    os.makedirs(final_path, exist_ok=True)
    model.base_model.base_model.save_pretrained(final_path)
    
    logger.info(f"Weights written to {best_path} and {final_path}")

def run_backtest():
    logger.info("=== STARTING PHASE 3: HISTORICAL SIMULATION ===")
    
    with open("configs/kairos_lora_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    logger.info("Loading pre-computed test tensors...")
    X_windows = torch.load("data/embeddings/X_windows_mtf_raw.pt")
    Y_alpha = torch.load("data/embeddings/Y_alpha_mtf.pt")
    Y_cov = torch.load("data/embeddings/Y_cov_mtf.pt")

    split_idx = int(len(X_windows) * 0.8)
    
    X_test = X_windows[split_idx:]
    Y_alpha_test = Y_alpha[split_idx:]
    Y_cov_test = Y_cov[split_idx:]
    
    test_dataset = TensorDataset(X_test, Y_alpha_test, Y_cov_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    logger.info("Loading best LoRA weights for out-of-sample evaluation...")
    model = load_base_model()
    
    adapter_path = "data/models/kairos_lora_weights_best"
    if os.path.exists(adapter_path):
        model.base_model.base_model.load_adapter(adapter_path, "default")
        logger.info("Successfully loaded trained LoRA adapters for backtest.")
    else:
        logger.error(f"Adapter weights not found at {adapter_path}. Aborting backtest.")
        return

    model = model.to("mps")
    model.eval()

    backtester = BacktestEngine(model, config, device="mps")
    logger.info(f"Initiating sequential backtest over {len(X_test)} historical steps...")
    
    results = backtester.run(test_loader)

async def run_live_trader():
    logger.info("=== STARTING PHASE 3: LIVE EXECUTION ENGINE ===")
    
    with open("configs/kairos_lora_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    paper_trading = config.get("PAPER_TRADING", True)
    if not paper_trading and not os.getenv("BINANCE_TESTNET_API_KEY"):
        logger.error("BINANCE_TESTNET_API_KEY is missing in .env for live exchange mode!")
        
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", 
               "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT"]
               
    logger.info(f"Loading base model and trained LoRA weights (Mode: {'Paper Trading' if paper_trading else 'Live Exchange'})...")
    model = load_base_model()
    
    adapter_path = "data/models/kairos_lora_weights_best"
    if os.path.exists(adapter_path):
        model.base_model.base_model.load_adapter(adapter_path, "default")
        logger.info("Successfully loaded trained LoRA adapters.")
    else:
        logger.warning(f"No trained weights found at {adapter_path}. Operating on base weights.")
        
    model = model.to("mps")
    model.eval()
    
    engine = LiveExecutionEngine(model=model, config=config, symbols=symbols)
    
    try:
        await engine.start()
        while engine.is_trading:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logger.warning("Async loop cancelled by system.")
    except Exception as e:
        logger.critical(f"Fatal error in live trader: {e}", exc_info=True)
    finally:
        logger.info("Initiating graceful shutdown sequence...")
        await engine.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kairos Multi-Asset Crypto Trading Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "backtest", "live"],
                        help="Execution mode: train (Phase 1), backtest (Simulation), or live (Execution)")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "train":
            run_training()
        elif args.mode == "backtest":
            run_backtest()
        elif args.mode == "live":
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                
            try:
                asyncio.run(run_live_trader())
            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt detected. Shutting down live pipeline.")
    except Exception as fatal_e:
        logger.critical(f"Unhandled top-level exception: {fatal_e}", exc_info=True)
        sys.exit(1)