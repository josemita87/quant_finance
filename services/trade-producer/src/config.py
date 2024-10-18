from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List, Literal

# Define a Pydantic settings class. This class is environment-agnostic 
# i.e, it does not care on which environment it is running.
# It will validate the configuration values and raise an error if they are not valid.

class Config(BaseSettings):
    kafka_broker_address: str = Field(..., env='KAFKA_BROKER_ADDRESS') 
    kafka_topic_name: str = Field(..., env='KAFKA_TOPIC_NAME')
    
    product_ids: List[str] = Field(..., env='PRODUCT_IDS')  

    # Extra parameters needed when running the producer
    live_or_historical: Literal['live', 'historical'] = Field(..., env='LIVE_OR_HISTORICAL')  # Environment variable
    last_n_days: int = Field(..., env='LAST_N_DAYS')  # Environment variable, casted to int

    # Validator for product IDs
    @field_validator('product_ids', mode='before')
    def validate_product_id_format(cls, v):
        if isinstance(v, list):
            for item in v:
                if not isinstance(item, str) or '/' not in item:
                    raise ValueError('Each product ID must be a string in the format "X/Y"')
        else:
            raise ValueError('product_ids must be a list of strings')
        return v

    class Config:
        env_file = '.env'  # Optional: Specifies an environment file to load

# Example usage:
config = Config()
