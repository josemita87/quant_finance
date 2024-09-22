
from typing import List, Dict
import ssl
import certifi

# Create SSL context using certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())

from websocket import create_connection
import json
import time

class KrakenwebsocketTradeAPI:
    
    URL = 'wss://ws.kraken.com/v2'

    def __init__ (
        self,
        product_id:str,
    ):
        """
        Initialize the Kraken websocket API connection with a product id.

        :param product_id: the id of the product to subscribe to.
        """

        self.product_id = product_id

        #Connect to Kraken
        self._ws = create_connection(self.URL, sslopt={"context": ssl_context})
        print('Connection established with Kraken API.')

        #Subscribe to trades
        self._subscribe_to_trades()

    def _subscribe_to_trades(self) -> None:
        
        """
      
        When called, this method will send a message to the Kraken API to
        subscribe to trades for the given product. 

        :return: None
        """
        print(f'Subscribing to trades for {self.product_id}.')

        # Subscribe to trades
        msg = {
            'method': 'subscribe',
            'params': {
                'channel': 'trade',
                'symbol': [
                    self.product_id
                ],
                'snapshot': False 
            }
        }
        
        self._ws.send(json.dumps(msg))

        #Dump first two responses (metadata), as they dont contain trade data
        _ = self._ws.recv()
        _ = self._ws.recv()



    def get_trades(self) -> List[Dict]:

        """
        Retrieve a list of trades from the Kraken API.

        :return: a list of dictionaries, each containing the trade id, price, volume, and time.
        """     

        message = self._ws.recv()

        if 'heartbeat' in message:
            return []
        
        message = json.loads(message)
        
        trades = []

        for trade in message['data']:
            trades.append({
                'product_id': self.product_id,
                'price' : trade['price'],
                'quantity' : trade['qty'],
                'timestamp': trade['timestamp']
            })
     
        
        return trades