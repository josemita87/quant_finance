from pydantic_settings import BaseSettings
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

class Config(BaseSettings):
    hopswork_project_name:str = os.environ['HOPSWORK_PROJECT_NAME']
    hopswork_api_key:str = os.environ['HOPSWORK_API_KEY']
    kafka_broker_address:str = os.environ['KAFKA_BROKER_ADDRESS']

config = Config()