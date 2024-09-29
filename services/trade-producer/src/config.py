import os 
from dotenv import load_dotenv, find_dotenv
from pydantic import validator, Field
from pydantic_settings import BaseSettings
from typing import List, Literal
load_dotenv(find_dotenv())


class Config(BaseSettings):

    kafka_broker_address: str = os.environ['KAFKA_BROKER_ADDRESS']
    kafka_topic_name: str = 'trade'
    product_ids: List[str] = Field(
        default=['ETH/EUR', 'BTC/EUR'], 
        description="List of product IDs in 'X/Y' format"
    )

    #Extra parameters needed when running the producer
    live_or_historical: Literal['live', 'historical'] = 'historical'
    last_n_days:int = 7
    
    @validator('product_ids', each_item=True)
    def validate_product_id_format(cls, v):
        if not isinstance(v, str) or '/' not in v:
            raise ValueError('Each product ID must be a string in the format "X/Y"')
        return v
    
config = Config()