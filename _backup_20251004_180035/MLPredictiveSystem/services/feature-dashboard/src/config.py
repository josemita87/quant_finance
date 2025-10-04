from pydantic import Field
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    project_name: str = Field(..., json_schema_extra={'env': 'PROJECT_NAME'})
    feature_group_name: str = Field(..., json_schema_extra={'env': 'FEATURE_GROUP_NAME'})
    feature_group_version: int = Field(..., json_schema_extra={'env': 'FEATURE_GROUP_VERSION'})
    feature_view_version: int = Field(..., json_schema_extra={'env': 'FEATURE_VIEW_VERSION'})
    feature_view_name: str = Field(..., json_schema_extra={'env': 'FEATURE_VIEW_NAME'})
    api_key: str = Field(..., json_schema_extra={'env': 'API_KEY'})

config = Config()

