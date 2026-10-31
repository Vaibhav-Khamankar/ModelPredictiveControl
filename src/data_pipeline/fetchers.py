import os
import asyncio
from typing import List, Callable, Dict, Any
from binance import AsyncClient, BinanceSocketManager
from src.utils.logger import logger

class BinanceDataFetcher:
    """
    Handles secure asynchronous REST and WebSocket connections to the Binance USD-M Futures API.
    Strictly isolated from trade execution to maintain least-privilege security.
    """
    def __init__(self, use_testnet: bool = True):
        self.use_testnet = use_testnet
        self.client = None
        self.bsm = None
        self._active_sockets = []

    async def initialize(self) -> None:
        """
        Initializes the AsyncClient using the read-only API credentials injected from the environment.
        """
        prefix = "BINANCE_TESTNET" if self.use_testnet else "BINANCE_DATA"
        api_key = os.environ.get(f"{prefix}_API_KEY")
        api_secret = os.environ.get(f"{prefix}_API_SECRET")

        if not api_key or not api_secret:
            logger.critical(f"Missing {prefix} credentials. Cannot initialize data ingestion.")
            raise ValueError(f"Environment variables for {prefix} missing.")

        # Instantiate for USD-M Futures
        self.client = await AsyncClient.create(
            api_key=api_key,
            api_secret=api_secret,
            testnet=self.use_testnet
        )
        self.bsm = BinanceSocketManager(self.client)
        logger.info(f"Binance AsyncClient initialized. Testnet: {self.use_testnet}")

    async def fetch_historical_futures_klines(self, symbol: str, interval: str, limit: int = 1000) -> List[Any]:
        """
        Fetches historical OHLCV data via REST for the Phase 1 pre-computation.
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Call initialize() first.")
            
        logger.debug(f"Fetching {limit} historical {interval} klines for {symbol}...")
        try:
            # futures_klines is specific to the USD-M futures API
            klines = await self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            return klines
        except Exception as e:
            logger.error(f"Failed to fetch historical klines for {symbol}: {e}")
            raise

    async def start_multiplex_stream(self, symbols: List[str], interval: str, callback: Callable[[Dict], None]) -> None:
        """
        Opens a single multiplexed WebSocket stream for OHLCV, Open Interest, and Mark Price (Funding).
        """
        if not self.bsm:
            raise RuntimeError("BinanceSocketManager not initialized.")

        streams = []
        for sym in symbols:
            s = sym.lower()
            streams.append(f"{s}@kline_{interval}")
            streams.append(f"{s}@markPrice")     # Contains real-time funding rate
            streams.append(f"{s}@openInterest")  # Contains open interest data

        logger.info(f"Starting multiplex WebSocket for {len(symbols)} assets ({len(streams)} streams).")
        
        # Multiplex socket combines streams to avoid hitting the 300 connection limit
        ts = self.bsm.multiplex_socket(streams)
        
        async with ts as tscm:
            self._active_sockets.append(tscm)
            while True:
                try:
                    res = await tscm.recv()
                    if res:
                        # Offload message to the async queue callback (in live_trader.py)
                        callback(res)
                except Exception as e:
                    logger.warning(f"WebSocket multiplex stream error: {e}. Attempting recovery...")
                    await asyncio.sleep(1)


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
    
    async def get_positions(self) -> dict:
        """Fetches current open positions using python-binance."""
        try:
            positions = await self.client.futures_position_information()
            pos_dict = {}
            for p in positions:
                sym = p.get('symbol')
                amt = float(p.get('positionAmt', 0.0))
                if sym in self.symbols and amt != 0.0:
                    pos_dict[sym] = amt
            return pos_dict
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            raise

    async def close(self) -> None:
        """
        Gracefully terminates the asynchronous client session.
        """
        if self.client:
            await self.client.close_connection()
            logger.info("Binance AsyncClient connection closed.")