# Kairos Crypto Model Predictive Control

A research-oriented, multi-asset cryptocurrency forecasting and trading pipeline. It adapts the pretrained Kairos 50M time-series model with LoRA, predicts asset returns and portfolio risk, and supports model training, historical backtesting, and asynchronous Binance execution.

> **Risk warning:** This software is experimental and is not financial advice. Cryptocurrency and leveraged futures trading can result in substantial or total loss. Backtest performance does not predict future results. Review the code and exchange permissions before connecting an account. Live execution is disabled by default, but must be explicitly kept in paper mode and carefully checked before use.

## What the project does

The pipeline is organized around these stages:

1. **Prepare market data** — download aligned 5-minute Binance USD-M Futures candles and build time-series training tensors.
2. **Forecast returns and risk** — use a frozen Kairos foundation-model trunk with trainable LoRA adapters and prediction heads for expected returns and a positive-semidefinite covariance matrix.
3. **Allocate a portfolio** — use a CVXPY mean-variance optimizer with configured allocation bounds, reserve cash, and turnover costs.
4. **Evaluate or execute** — run a sequential historical simulation or consume Binance market streams in paper/live mode.

The command-line entry point currently exposes training, backtesting, and live execution. Data preparation is performed separately with scripts under `scripts/`.

## Assets and input data

The pipeline is configured for these ten USDT pairs:

`BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `ADAUSDT`, `AVAXUSDT`, `DOGEUSDT`, `DOTUSDT`, and `LINKUSDT`.

Each asset uses seven features in this order:

| Feature | Description |
| --- | --- |
| `open` | Candle open price |
| `high` | Candle high price |
| `low` | Candle low price |
| `close` | Candle close price |
| `volume` | Candle volume |
| `open_interest` | Open interest |
| `funding_rate` | Funding rate |

The model config declares a 96-candle context. The current multi-timeframe builder concatenates 96 five-minute rows, 24 hourly rows, and 14 daily rows, producing 134 time steps per sample (with seven feature values for each of ten assets). It also builds expected-return, covariance, and next-step-return targets.

**Data caveat:** The current history-download and live tensor-building paths fill open interest and funding-rate features with zeros. Although the feature slots are part of the model input, these paths do not currently populate them with observed values.

## Architecture

```text
Binance history / WebSocket market data
                  |
       data preparation scripts
                  |
    134-step, 10-asset tensors
                  |
 frozen Kairos 50M trunk + LoRA adapters
                  |
   +--------------+---------------+
   |                              |
 expected-return head       covariance head
                                  |
                  mean-variance optimizer
                  /                     \
        historical backtest        execution engine
                                  paper or live mode
```

The covariance prediction head outputs a lower-triangular factor and constructs the covariance matrix as `L @ L.T`, ensuring a symmetric positive-semidefinite result. The portfolio optimizer applies constraints from `configs/optimization_config.yaml`, including a 25% maximum weight per asset and a 10% reserve-cash target under the default settings.

## Repository layout

```text
.
├── main.py                         # CLI entry point: train, backtest, live
├── configs/
│   ├── kairos_lora_config.yaml     # Model, training, and trading settings
│   ├── optimization_config.yaml    # Portfolio constraints and penalties
│   └── backtest_config.yaml        # Backtest-related settings
├── scripts/
│   ├── download_history.py         # Download 5-minute futures candles
│   ├── build_dataset.py            # Build aligned base tensors
│   └── build_mtf.py                # Build multi-timeframe tensors for main.py
├── src/
│   ├── data_pipeline/              # Binance ingestion and tensor builders
│   ├── models/                     # Kairos wrapper, LoRA, forecasting heads
│   ├── optimizer/                  # CVXPY portfolio optimization
│   ├── engine/                     # Training, backtesting, live execution
│   └── utils/                      # Logging and metrics
├── tests/                          # Shape, covariance, and lookahead tests
└── requirements.txt
```

Generated market data, tensor files, and model weights are stored under `data/` and are excluded from Git by `.gitignore`. The data preparation scripts are included; generated artifacts must be created locally before running the relevant modes.

## Requirements and setup

The model and entry point are configured to use PyTorch's **MPS** device, so the primary runtime target is an Apple Silicon Mac with a compatible PyTorch installation. Python 3.11 is a suitable starting point for the current project environment.

```bash
git clone <your-repository-url>
cd ModelPredictiveControl

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The source also imports `peft` and `tsfm.model.kairos`; neither is currently declared in `requirements.txt`. Make sure the environment provides compatible packages for those imports and the Kairos model before launching `main.py`. The first model load may also download the pretrained `mldi-lab/Kairos_50m` model from Hugging Face.

Kairos is loaded with `trust_remote_code=True`. Review the model repository's code and security implications before running it.

## Prepare training and backtest data

The main training and backtest modes expect the following precomputed tensors:

```text
data/embeddings/X_windows_mtf_raw.pt
data/embeddings/Y_alpha_mtf.pt
data/embeddings/Y_cov_mtf.pt
```

Generate them from historical candles with:

```bash
python scripts/download_history.py
python scripts/build_mtf.py
```

The downloader fetches recent 5-minute futures candles for the configured ten assets and saves CSVs under `data/raw/`. The multi-timeframe builder reads those files and writes the tensors under `data/embeddings/`. These data files are local artifacts, not tracked in the repository.

`scripts/build_dataset.py` is an alternative aligned-dataset builder that writes tensors under `data/processed/`; those filenames are **not** the input paths used by the `train` and `backtest` modes in `main.py`.

## Run the application

From the project root, after installing dependencies and preparing the required artifacts:

```bash
python main.py --mode train
python main.py --mode backtest
python main.py --mode live
```

### Training

Training uses the first 80% of the precomputed tensor windows and validates on the final 20%. Epoch count, batch size, learning rate, and other settings are read from `configs/kairos_lora_config.yaml`. The best and final LoRA adapter weights are saved to:

```text
data/models/kairos_lora_weights_best/
data/models/kairos_lora_weights_final/
```

### Backtesting

Backtesting uses the final 20% of the same precomputed tensors as a sequential holdout set and loads the best adapter weights. It reports simulated portfolio and benchmark metrics through the project logger. Run training first so `data/models/kairos_lora_weights_best/` exists.

### Paper and live execution

`configs/kairos_lora_config.yaml` contains the execution controls:

```yaml
USE_TESTNET: false
PAPER_TRADING: true
```

Keep `PAPER_TRADING: true` unless you have deliberately reviewed and configured live execution. In paper mode, the engine still connects to Binance to seed market history and receive WebSocket data; paper mode does not mean offline mode. The engine is configured with `USE_TESTNET: false` by default, so check that setting and the corresponding account/API permissions before starting it.

The fetcher reads API credentials from environment variables. With `USE_TESTNET: false`, its current prefix is `BINANCE_DATA`, so it expects:

```dotenv
BINANCE_DATA_API_KEY=your_read_only_market_data_key
BINANCE_DATA_API_SECRET=your_read_only_market_data_secret
```

With testnet enabled, the fetcher expects `BINANCE_TESTNET_API_KEY` and `BINANCE_TESTNET_API_SECRET`. Keep secrets in a local `.env` file (ignored by Git); never commit API keys. Use the least-privileged keys possible. The live-mode entry point has an additional testnet-key check when paper trading is disabled, so verify that check and the fetcher's configured credential prefix agree before using live execution.

The engine writes JSON Lines telemetry under `logs/telemetry/` by default. Review `configs/kairos_lora_config.yaml` and the live execution code before changing paper mode or risk controls.

## Configuration

| File | Purpose |
| --- | --- |
| `configs/kairos_lora_config.yaml` | Base model dimensions, input features, LoRA parameters, training settings, paper/live controls, and telemetry location |
| `configs/optimization_config.yaml` | Mean-variance objective, exposure and asset bounds, reserve cash, and transaction-cost penalties |
| `configs/backtest_config.yaml` | Backtest-oriented configuration values and reporting preferences |

The training and trading entry-point modes load `kairos_lora_config.yaml`. The live optimizer separately reads `optimization_config.yaml`. Check the implementation before assuming every setting in `backtest_config.yaml` is active in the current `main.py` backtest path.

## Tests

Run the test suite from the repository root:

```bash
python -m pytest
```

The tests cover forecast output shapes and covariance-matrix properties, tensor-window behavior, and portfolio constraint bounds.

## Limitations

- Results depend on the quality, alignment, and representativeness of the historical data and targets.
- The default historical downloader uses zero-valued open-interest and funding-rate fields rather than fetching those observations.
- MPS is selected explicitly in the main model/training path; running that path on a machine without compatible MPS support may require code/configuration changes.
- A missing best adapter does not necessarily stop live mode: the engine logs a warning and can proceed with base weights. Do not treat that as a trained strategy.
- Paper trading and backtesting are simulations, not guarantees of exchange execution quality or future returns.

## License

No license file is currently included. Add a license before redistributing this project or accepting external contributions.
