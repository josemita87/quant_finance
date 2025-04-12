from typing import List, Dict, Tuple, Optional
from src.kraken_api.trade import Trade
from datetime import timezone, datetime
from time import sleep
from loguru import logger
import requests
from pathlib import Path
import pandas as pd


class KrakenRestAPI:

    def __init__(
        self,
        product_ids: List[str],
        last_n_days: int,
        cache_dir: Optional[str] = None
    )-> None:
        """
        Initialize the Kraken API REST client.
        Params:
            product_id (str): Product ID to fetch data for.
            last_n_days (int): Number of days in the past to fetch data from.
            cache_dir_path (Optional[str]): Path to the cache directory.
        Attributes:
            to_ms (int): Current date in milliseconds since epoch.
            from_ms (int): Start date in milliseconds since epoch, calculated based on `last_n_days`.
            since_ns (int): Start date in nanoseconds since epoch.
            last_trade_ms (int): Timestamp of the last trade in milliseconds since epoch.
            product_id (str): Product ID to fetch data for.
            is_finished (bool): Flag indicating whether the data fetching is finished.
        """
        
        self.ids_to_process = [id for id in product_ids]
        self.current_product_id = self.ids_to_process[0]
        self.is_finished = False


        #Initialize cache directory
        if cache_dir:
            self.cache = CachedTrades(cache_dir)
            self.use_cache = True

        #Static params (only updated in the constructor)
        self.to_ms, self.from_ms  = self.time_related_params(last_n_days)

        #This attribute will be updated through the process. Initially set to the start date
        self.last_trade_ms = self.from_ms
        
        
    def time_related_params(self, last_n_days: int) -> Tuple[int, int]:
        
        today_date = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        # Current date in milliseconds since epoch
        to_ms = int(today_date.timestamp() * 1000)
        # Start date in milliseconds since epoch
        from_ms = to_ms - last_n_days * 24 * 60 * 60 * 1000

        return to_ms, from_ms
    

    def get_trades(self) -> List[Trade]:
        
        if self.use_cache:
            trades: List[Trade] = self.cache.read(self.last_trade_ms, self.current_product_id)
          
            #In case the cache contained data, return the trades
            if trades:
                self.last_trade_ms = trades[-1].timestamp_ms + 1
                logger.info(f'Cached {len(trades)} trades {self.current_product_id}')
                
                return trades
            
        #If cache returned None, fetch from the API
        trades : List[Trade] = self.fetch_trades_from_api()
        logger.info(f'APICALL {len(trades)} trades {self.current_product_id}')
        
        #Store trades to a new cache file
        if self.use_cache:
            self.cache.write(self.last_trade_ms, self.current_product_id, trades)
            logger.debug(self.last_trade_ms)

        #Update the last trade timestamp (AFTER HAVING WRITTEN CACHE, IF NOT CACHE WOULD BE MISPLACED)
        self.last_trade_ms = trades[-1].timestamp_ms + 1

        #When no currencies remaining, we are done
        self.is_finished = not self.ids_to_process
        
        return trades
        

    def fetch_trades_from_api(self)-> List[Trade]:
          
        payload = {}
        headers = {'Accept': 'application/json'}
        url = f'https://api.kraken.com/0/public/Trades?pair={self.current_product_id}&since={self.last_trade_ms//1000}'

        data = requests.get(url, params=payload, headers=headers).json()
        
        if ('error' in data) and \
        ('EGeneral:Too many requests' == data['error']):
            logger.info('Too many requests. Waiting for 30 seconds')
            sleep(30)
       
        trades = [
            Trade(
                product_id=self.current_product_id,
                price=trade[0],
                volume=trade[1],
                timestamp_ms=int(trade[2] * 1000) 
            )
            for trade in data['result'][self.current_product_id]
        ]

        #Filter out trades that are after the end timestamp
        trades = [trade for trade in trades if trade.timestamp_ms <= self.to_ms]

        #In here, we were setting self.last_trade_ms to int(data['result']['last']) but it was wrong, 
        #Notice how in trades, our last trade is the last trade in the list before self.to_ms!!!!
        #Thus, there could be trades in between the last trade and self.to_ms that we are missing.

        
        if int(data['result']['last']) // 1000000 >= self.to_ms:   
            logger.debug(f'Finished processing {self.ids_to_process.pop(0)}')
            if len(self.ids_to_process) > 0:
                logger.debug(f'Next currency to process: {self.ids_to_process[0]}')

            #Restart the process with the next currency (time window is reset)
            self.current_product_id = self.ids_to_process[0]
            self.last_trade_ms = self.from_ms

        sleep(1)
        return trades
  


class CachedTrades():
    """ A class to cache the trades fetched from the Kraken API.
    Based on the product_id and start date, the trades are cached in a 
    parquet filein the specified directory (cache_dir)."""

    def __init__(self, cache_dir: str) -> None:

        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)

    
    def read(self, since_ms:int, product_id) -> List[Trade]|None:
        """
        Read the cached trades from the file.
        Params:
            product_id (str): Product ID to fetch data for.
            since_sec (int): Start date in seconds since epoch.
        Returns:
            List[Trade]: List of trades.
        """
        file_path = self._get_file_path(since_ms, product_id)
        
        if file_path.exists():
            data = pd.read_parquet(file_path)
            logger.debug(f'TRYING TO READ FOR filepath {file_path}')
            
            return [Trade(**trade) for trade in data.to_dict(orient='records')]
        else:
            logger.warning(f'File does not exist: {file_path}')
            return None
        
        
    

    def write(self, since_ms: int, product_id: str, trades: List[Trade]) -> None:
        """
        Append the new trades to the existing cached file or create a new file.
        Params:
            since_sec (int): Start date in seconds since epoch.
            trades (List[Trade]): List of trades to be cached.
        """
        file_path = self._get_file_path(since_ms, product_id)
        
        # Convert the list of trades into a DataFrame
        data = pd.DataFrame([dict(trade) for trade in trades])
        
        #Write (or overwrite if existing file) records to the parquet file
        data.to_parquet(file_path)
        if file_path.exists():
            logger.debug(f'File successfully written: {file_path}')
        else:
            logger.warning(f'File was not created: {file_path}')
        



    def _get_file_path(self, since_nsec: int, product_id: str) -> Path:
        """
        Generate a file path for the cached trades.
        Params:
            since_nsec (int): Start time in nanoseconds since epoch.
        Returns:
            Path: The full path of the cache file.
        """
        # Create a unique file name based on product_id and since_nsec
        file_name = f"{product_id.replace("/", "_")}_{since_nsec}.parquet"
        # Return the full path by combining cache_dir and the file name
        return self.cache_dir / file_name