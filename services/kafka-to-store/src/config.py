from pydantic_settings import BaseSettings
from dotenv import load_dotenv, find_dotenv
from typing import Optional
from pydantic import Field

# Load environment variables from a .env file
load_dotenv(find_dotenv())

class Config(BaseSettings):
    #Kafka configuration
    kafka_broker_address: str = Field(..., env='KAFKA_BROKER_ADDRESS')
    kafka_input_topic: str = Field(..., env='KAFKA_INPUT_TOPIC')
    kafka_consumer_group: str = Field(..., env='KAFKA_CONSUMER_GROUP')
    auto_offset_reset: str = Field(..., env='AUTO_OFFSET_RESET')

    #Hopsworks configuration
    hopswork_project_name: str = Field(..., env='HOPSWORK_PROJECT_NAME')
    hopswork_feature_group_name: str = Field(..., env='FEATURE_GROUP_NAME')
    hopswork_group_version: str = Field(..., env='HOPSWORK_GROUP_VERSION')
    hopswork_api_key: str = Field(..., env='HOPSWORK_API_KEY')
    online_offline: str = Field(..., env='ONLINE_OFFLINE')
    
    #Function parameters
    buffer_size: int = Field(..., env='BUFFER_SIZE')
    timer: Optional[int] = Field(None, env='TIMER')  

# Instantiate the config
config = Config()
