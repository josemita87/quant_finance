from config import config

from quixstreams import Application
from loguru import logger
import hopswork_api
import json

def kafka_to_feature_store(
    kafka_topic: str,
    kafka_broker: str,
    feature_group_name: str,
    feature_group_version:str
) -> None:
    """
    Function to transfer data from Kafka to a feature store.

    Args:
        kafka_topic (str): The Kafka topic to read from.
        kafka_broker (str): The Kafka broker address.
        feature_group_name (str): The name of the feature group.
        feature_group_version (str): The version of the feature group.

    Returns:
        None
    """

    app = Application(
        broker_address=kafka_broker, 
        consumer_group='kafka-to-feature-store'
    )
    print(f"Connected to Kafka broker at {kafka_broker}")
    #input = app.topic(name=kafka_topic, value_deserializer='json')
    
    with app.get_consumer() as consumer:

        consumer.subscribe(topics = [kafka_topic])
        
        while True:
            msg = consumer.poll(1)
        
            if not msg:
                #logger.info("No new messages")
                continue
            elif msg.error():
                logger.error(f"Consumer error: {msg.error()}")
                continue

            else:
                ohlc = json.loads(msg.value().decode('utf-8'))
                logger.info(ohlc)
                
                # Write the data to the feature store
                hopswork_api.push_data_to_feature_store(
                    feature_group_name=feature_group_name,
                    feature_group_version=feature_group_version,
                    data=ohlc
                )

if __name__ == '__main__':  
    kafka_to_feature_store(
        kafka_topic='ohlc',
        kafka_broker= config.kafka_broker_address,
        feature_group_name='ohlc',
        feature_group_version='1'
    )