from quixstreams import Application
from typing import List, Dict

def produce_trades(
    kafka_broker: str,
    kafka_topic_name: str
    )-> None:

    """
    Produce a stream of random trades to a Kafka topic.

    :param kafka_broker: The address of the Kafka broker.
    :param kafka_topic: The name of the Kafka topic to write to.
    """

    app = Application(broker_address= kafka_broker)
    topic = app.topic(name = kafka_topic_name, value_serializer = 'json')

    from src.kraken_api import KrakenwebsocketTradeAPI
    kraken_api = KrakenwebsocketTradeAPI(product_id = 'BTC/USD')

    print('Connecting to Kraken API...')

    while True:

        trades: List[Dict] = kraken_api.get_trades()
        print('Trades received')

        for trade in trades:
            print(f"Trade size: {len(str(trade).encode('utf-8'))} bytes")    
            import time
            time.sleep(10)
            with app.get_producer() as producer:
                
                message = topic.serialize(key = trade["product_id"], value = trade)

                # Produce a messace into the kafka topic.
                producer.produce(
                    topic=topic.name, 
                    value=message.value, 
                    key=message.key
                )

            from time import sleep
            print('message sent')
            sleep(1)
            

if __name__ == '__main__':
    produce_trades(
        'redpanda-0:9092',
        'trade'
    )