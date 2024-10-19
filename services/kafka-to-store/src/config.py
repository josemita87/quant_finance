from pydantic_settings import BaseSettings
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

class Config(BaseSettings):
    kafka_input_topic:str = os.environ['KAFKA_INPUT_TOPIC']
    kafka_broker_address:str = os.environ['KAFKA_BROKER_ADDRESS']
    kafka_consumer_group:str = os.environ['KAFKA_CONSUMER_GROUP']
    auto_offset_reset:str = os.environ['AUTO_OFFSET_RESET']
    hopswork_project_name:str = os.environ['HOPSWORK_PROJECT_NAME']
    hopswork_feature_group_name:str = os.environ['FEATURE_GROUP_NAME']
    hopswork_group_version:str = os.environ['HOPSWORK_GROUP_VERSION']
    hopswork_api_key:str = os.environ['HOPSWORK_API_KEY']
    
    online_offline:str = os.environ['ONLINE_OFFLINE']
    buffer_size:int = os.environ['BUFFER_SIZE']
    timer:int = os.environ['TIMER']
    
config = Config()