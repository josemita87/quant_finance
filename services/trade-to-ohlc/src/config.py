from pydantic_settings import BaseSettings
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

class Config(BaseSettings):
    kafka_broker_address: str = os.environ['KAFKA_BROKER_ADDRESS']
    kafka_input_topic_name: str = os.environ['KAFKA_INPUT_TOPIC_NAME']
    kafka_output_topic_name: str = os.environ['KAFKA_OUTPUT_TOPIC_NAME']
    ohlc_window_seconds: int = os.environ['OHLC_WINDOW_SECONDS']
    consumer_group: str = 'trade-to-ohlc-backfill'
config = Config()