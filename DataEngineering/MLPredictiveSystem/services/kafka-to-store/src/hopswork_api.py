
import hopsworks
import pandas as pd
from config import config
from loguru import logger


def push_data_to_feature_store(
    feature_group_name: str,
    feature_group_version: str,
    data: list[dict],
    online_offline: str
) -> None:
    """
    Function to push data to a feature store.

    Args:
        feature_group_name (str): The name of the feature group.
        feature_group_version (str): The version of the feature group.
        data (dict): The data to push to the feature store.
        online_offline(str): save to online of offline feature group.
    Returns:
        None
    """

   
    project = hopsworks.login(
        project = config.hopswork_project_name,
        api_key_value = config.hopswork_api_key
    )

    feature_store = project.get_feature_store()

    ohlc_fg = feature_store.get_or_create_feature_group(
        name=feature_group_name,
        version=feature_group_version,
        primary_key=['product_id', 'timestamp'],
        event_time= 'timestamp',
        online_enabled=True
    )

    data = pd.DataFrame(data)
    
    ohlc_fg.insert(
        data, write_options = {
            'start_offline_materialization': True if online_offline == 'offline' else False
        }
    )