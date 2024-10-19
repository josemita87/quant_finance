from pydantic_settings import BaseSettings
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

class Config(BaseSettings):
    kafka_broker_address: str = os.environ['KAFKA_BROKER_ADDRESS']
    kafka_input_topic_name: str = os.environ['KAFKA_INPUT_TOPIC']
    kafka_output_topic_name: str = os.environ['KAFKA_OUTPUT_TOPIC']
    ohlc_window_seconds: int = os.environ['OHLC_WINDOW_SECONDS']
    consumer_group: str = os.environ['CONSUMER_GROUP']
    auto_offset_reset: str = os.environ['AUTO_OFFSET_RESET']

config = Config()