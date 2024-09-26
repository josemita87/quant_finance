from typing import Dict, List
from src.config import config
from loguru import logger
from quixstreams import Application


def produce_trades(kafka_broker: str, kafka_topic_name: str, product_id: str) -> None:
    """
    Produce a stream of random trades to a Kafka topic.

    :param kafka_broker: The address of the Kafka broker.
    :param kafka_topic: The name of the Kafka topic to write to.
    """

    app = Application(broker_address=kafka_broker)
    topic = app.topic(name=kafka_topic_name, value_serializer='json')

    from src.kraken_api import KrakenwebsocketTradeAPI

    kraken_api = KrakenwebsocketTradeAPI(product_id=product_id)

    logger.info('Connecting to Kraken API...')

    while True:
        logger.info('Waiting for trades ...')

        trades: List[Dict] = kraken_api.get_trades()

        for trade in trades:

            with app.get_producer() as producer:
                message = topic.serialize(key=trade['product_id'], value=trade)

                # Produce a messace into the kafka topic.
                producer.produce(topic=topic.name, value=message.value, key=message.key)
                logger.info(message.value)

            from time import sleep
            sleep(1)


if __name__ == '__main__':

    produce_trades(
        config.kafka_broker_address, 
        config.kafka_topic_name,
        config.product_id
    )
