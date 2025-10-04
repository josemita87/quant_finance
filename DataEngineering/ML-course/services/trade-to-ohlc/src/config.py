from pydantic_settings import BaseSettings
from dotenv import load_dotenv, find_dotenv
from pydantic import Field

# Load environment variables from .env file
load_dotenv(find_dotenv())

class Config(BaseSettings):
    kafka_broker_address: str = Field(..., env='KAFKA_BROKER_ADDRESS')
    kafka_input_topic: str = Field(..., env='KAFKA_INPUT_TOPIC')
    kafka_output_topic: str = Field(..., env='KAFKA_OUTPUT_TOPIC')
    auto_offset_reset: str = Field(..., env='AUTO_OFFSET_RESET')
    ohlc_window_seconds: int = Field(..., env='OHLC_WINDOW_SECONDS')
    consumer_group: str = Field(..., env='CONSUMER_GROUP')

# Initialize config with environment variables
config = Config()
