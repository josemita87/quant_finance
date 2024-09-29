from pydantic_settings import BaseSettings
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

class Config(BaseSettings):
    kafka_topic:str = os.environ['KAFKA_TOPIC']
    kafka_broker_address:str = os.environ['KAFKA_BROKER_ADDRESS']
    kafka_consumer_group:str = os.environ['KAFKA_CONSUMER_GROUP']
    hopswork_project_name:str = os.environ['HOPSWORK_PROJECT_NAME']
    hopswork_group_version:str = os.environ['HOPSWORK_GROUP_VERSION']

    hopswork_api_key:str = os.environ['HOPSWORK_API_KEY']
    hopswork_project_name:str = os.environ['HOPSWORK_PROJECT_NAME']
    

config = Config()