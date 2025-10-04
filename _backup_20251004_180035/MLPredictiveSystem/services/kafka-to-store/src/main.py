from config import config
from datetime import datetime, timezone
from quixstreams import Application
from loguru import logger
import hopswork_api
import json



def get_current_utc_secs() -> int:
    return int(datetime.now(timezone.utc).timestamp())

def kafka_to_feature_store(
    kafka_input_topic: str,
    kafka_broker: str,
    feature_group_name: str,
    feature_group_version:str,
    online_offline:str,
    timer:int,
    buffer_size:int,
    auto_offset_reset:str
) -> None:
    """
    Function to transfer data from Kafka to a feature store.

    Args:
        kafka_topic (str): The Kafka topic to read from.
        kafka_broker (str): The Kafka broker address.
        feature_group_name (str): The name of the feature group.
        feature_group_version (str): The version of the feature group.
        online_offline(str): save to online of offline feature store,
        this is for live, historical respectively.

    Returns:
        None
    """

    app = Application(
        broker_address=kafka_broker, 
        consumer_group=config.kafka_consumer_group,
        auto_offset_reset=auto_offset_reset
    )

    input_topic = app.topic(name=kafka_input_topic, value_deserializer='json', key_deserializer='string')
    buffer = []
    # Save the last time the data was saved to the feature store
    last_save_to_fs = get_current_utc_secs()
    
    logger.info(f"Connected to Kafka broker at {kafka_broker}")
    
    with app.get_consumer() as consumer:

        consumer.subscribe(topics = [input_topic.name])
        
        while True:
            msg = consumer.poll(1)
            

            if not msg:
                # If no new messages incoming in backfill mode, 
                # break the loop and push residual data 
                if online_offline == 'offline':
                        
                    if (get_current_utc_secs() - last_save_to_fs) > timer:
                
                        #avoid pushing empty data to feature store
                        if buffer:
                            hopswork_api.push_data_to_feature_store(
                                feature_group_name=feature_group_name,
                                feature_group_version=feature_group_version,
                                data=buffer,
                                online_offline=online_offline
                            )
                            last_save_to_fs = get_current_utc_secs()
            
                            break
                else:
                    continue

            elif msg.error():
                logger.error(f"Consumer error: {msg.error()}")
                continue

            else:
                
                ohlc = json.loads(msg.value().decode('utf-8'))
                buffer.append(ohlc)
                logger.info(ohlc)
                if len(buffer) >= buffer_size:
                    # Write the data to the feature store
                    logger.info(f"Writing {len(buffer)} records to the feature store")
                    hopswork_api.push_data_to_feature_store(
                        feature_group_name=feature_group_name,
                        feature_group_version=feature_group_version,
                        data=buffer,
                        online_offline=online_offline
                    )
                    buffer = []



if __name__ == '__main__':  
    kafka_to_feature_store(
        kafka_input_topic=config.kafka_input_topic,
        kafka_broker= config.kafka_broker_address,
        feature_group_name=config.hopswork_feature_group_name,
        feature_group_version=config.hopswork_group_version,
        online_offline = config.online_offline,
        timer = config.timer,
        buffer_size = config.buffer_size,
        auto_offset_reset=config.auto_offset_reset
    )