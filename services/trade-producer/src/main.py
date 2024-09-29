from typing import Dict, List
from config import config
from loguru import logger
from quixstreams import Application


def produce_trades(
    kafka_broker: str, 
    kafka_topic_name: str, 
    product_ids: List[str],
    live_or_historical,
    last_n_days:int
) -> None:
    """
    Produce a stream of random trades to a Kafka topic.

    :param kafka_broker: The address of the Kafka broker.
    :param kafka_topic: The name of the Kafka topic to write to.
    """

    app = Application(broker_address=kafka_broker)
    topic = app.topic(name=kafka_topic_name, value_serializer='json')
    logger.info('Connecting to Kraken API...')

    if live_or_historical == 'live':
        from kraken_api.websocket import KrakenwebsocketTradeAPI
        kraken_api = KrakenwebsocketTradeAPI(product_id=product_ids[0])

    else:
        from kraken_api.rest import KrakenRestAPIMultipleProducts 
        kraken_api = KrakenRestAPIMultipleProducts(product_ids, last_n_days)
    
    while True:
        logger.info('Waiting for trades ...')
        trades: List[List[Dict]] = kraken_api.get_trades()
        logger.info(f'Fetched {len(trades)} trades.')
        
        for trade in trades:
        
            with app.get_producer() as producer:
                message = topic.serialize(key=trade['product_id'], value=trade)

                # Produce a message into the kafka topic.
                producer.produce(topic=topic.name, value=message.value, key=message.key)
                logger.info(message.value)

            from time import sleep
            sleep(1)


if __name__ == '__main__':

    produce_trades(
        config.kafka_broker_address, 
        config.kafka_topic_name,
        config.product_ids,
        config.live_or_historical,
        config.last_n_days
    )
