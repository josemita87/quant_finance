from typing import Dict, List
from src import config
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
        trades: List[Dict] = kraken_api.get_trades()
        logger.info('Trades received')

        for trade in trades:
            logger.info(f"Trade size: {len(str(trade).encode('utf-8'))} bytes")

            with app.get_producer() as producer:
                message = topic.serialize(key=trade['product_id'], value=trade)

                # Produce a messace into the kafka topic.
                producer.produce(topic=topic.name, value=message.value, key=message.key)


if __name__ == '__main__':

    produce_trades(
        config.kafka_broker_adress, 
        config.kafka_topic_name,
        config.product_id
    )
