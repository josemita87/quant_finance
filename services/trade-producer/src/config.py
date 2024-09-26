import os 
from dotenv import load_dotenv, find_dotenv
from pydantic_settings import BaseSettings

load_dotenv(find_dotenv())


class Config(BaseSettings):

    kafka_broker_address: str = os.environ['KAFKA_BROKER_ADDRESS']
    kafka_topic_name: str = 'trade'
    product_id: str = 'ETH/EUR'


config = Config()