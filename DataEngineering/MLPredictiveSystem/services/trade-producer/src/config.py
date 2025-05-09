from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List, Literal, Optional

# Define a Pydantic settings class. This class is environment-agnostic 
# i.e, it does not care on which environment it is running.
# It will validate the configuration values and raise an error if they are not valid.

class Config(BaseSettings):

    kafka_broker_address: str = Field(..., env='KAFKA_BROKER_ADDRESS') 
    kafka_output_topic: str = Field(..., env='KAFKA_OUTPUT_TOPIC')
    live_or_historical: Literal['live', 'historical'] = Field(..., env='LIVE_OR_HISTORICAL')  
    product_ids: List[str] = Field(..., env='PRODUCT_IDS')  

    # Backfill mode specific
    last_n_days: Optional[int] = Field(None, env='LAST_N_DAYS') 
    cache_dir_path: Optional[str] = Field(None, env='CACHE_DIR_PATH')

    # Validators 
    @field_validator('product_ids', mode='before')
    def validate_product_id_format(cls, v):
        if isinstance(v, list):
            for item in v:
                if not isinstance(item, str) or '/' not in item:
                    raise ValueError('Each product ID must be a string in the format "X/Y"')
        else:
            raise ValueError('product_ids must be a list of strings')
        return v

    
config = Config()
