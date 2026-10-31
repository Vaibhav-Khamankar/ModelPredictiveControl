import os
import time
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from src.utils.logger import logger

# Ensure the output directory exists
os.makedirs("data/raw", exist_ok=True)

def download_historical_klines():
    """
    Downloads 6 months of 5m OHLCV data from Binance USD-M Futures.
    Formats the data to match the 7-feature [O, H, L, C, V, OI, Funding] requirement.
    """
    # Initialize public client (No API keys needed for historical klines)
    client = Client()
    
    symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", 
        "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT"
    ]
    
    interval = Client.KLINE_INTERVAL_5MINUTE
    start_str = "6 months ago UTC"
    
    logger.info(f"Starting historical download for {len(symbols)} assets. Target: past 6 months, 5m interval.")

    for symbol in symbols:
        logger.info(f"Fetching {symbol}...")
        
        try:
            # Fetch futures klines (fapi)
            klines = client.futures_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str
            )
            
            if not klines:
                logger.warning(f"No data returned for {symbol}.")
                continue

            # Binance Kline format: 
            # [Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades, Taker buy base, Taker buy quote, Ignore]
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_asset_volume', 'num_trades', 
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Convert timestamp to datetime index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Keep only the required columns and convert to float32 for MPS memory efficiency
            df = df[['open', 'high', 'low', 'close', 'volume']].astype('float32')
            
            # Append zeroed columns for Open Interest and Funding Rate to match the 7-feature live tensor shape
            df['open_interest'] = 0.0
            df['funding_rate'] = 0.0
            
            # Drop any incomplete current candle
            df = df.iloc[:-1]
            
            save_path = f"data/raw/{symbol}_5m.csv"
            df.to_csv(save_path)
            
            logger.info(f"Saved {len(df)} rows for {symbol} -> {save_path}")
            
            # Respect Binance REST API weight limits (klines are heavily weighted)
            time.sleep(1.5)
            
        except BinanceAPIException as e:
            logger.error(f"Binance API Error on {symbol}: {e}")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Unexpected error on {symbol}: {e}")

    logger.info("Historical data download complete. Ready for Phase 1 tensor building.")

if __name__ == "__main__":
    download_historical_klines()