import ssl
from typing import Dict, List
from src.kraken_api.trade import Trade
from loguru import logger
import certifi

# Create SSL context using certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())

import json
from datetime import datetime
from websocket import create_connection


class KrakenwebsocketTradeAPI:
    URL = 'wss://ws.kraken.com/v2'

    def __init__(
        self,
        product_ids: List[str],
    ):
        """
        Initialize the Kraken websocket API connection with a product id.

        :param product_id: the id of the product to subscribe to.
        """

        self.product_ids = product_ids

        # Connect to Kraken
        self._ws = create_connection(self.URL, sslopt={'context': ssl_context})
        logger.info('Connection established with Kraken API.')

        # Subscribe to trades
        self._subscribe_to_trades()

        #Set finished flag to false (always) because we are live streaming
        self.is_finished = False

        
    def _subscribe_to_trades(self) -> None:
        """

        When called, this method will send a message to the Kraken API to
        subscribe to trades for the given product.

        :return: None
        """
        logger.info(f'Subscribing to trades for {self.product_ids}.')

        # Subscribe to trades
        msg = {
            'method': 'subscribe',
            'params': {
                'channel': 'trade',
                'symbol': self.product_ids,
                'snapshot': False,
            },
        }
        
        self._ws.send(json.dumps(msg))

        # Dump first two responses (metadata), as they dont contain trade data
        _ = self._ws.recv()
        _ = self._ws.recv()
        logger.info(_)

    def get_trades(self) -> List[Trade]:
        """
        Retrieve a list of trades from the Kraken API.

        :return: a list of dictionaries, each containing the trade id, price, volume, and time.
        """

        message = self._ws.recv()

        if 'data' not in message:
            return []

        message = json.loads(message)
        trades = []
        
        
        for trade in message['data']:
            # Parse the RFC3339 timestamp to a datetime object
            dt = datetime.strptime(
                trade['timestamp'], 
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            # Convert the datetime object to Unix timestamp in milliseconds
            unix_timestamp_ms = int(dt.timestamp() * 1000)

            trades.append(
                Trade(
                    product_id=trade['symbol'],
                    price=trade['price'],
                    volume=trade['qty'],
                    timestamp_ms= unix_timestamp_ms
                )
            )    
            
        return trades
