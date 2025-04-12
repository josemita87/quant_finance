
import pandas as pd
import hopsworks
from src.config import config

def get_features_from_store() -> pd.DataFrame:
    """
    Retrieve the features from the feature store.
    """
    # Connect to the feature store
    project = hopsworks.login(
        project=config.project_name,
        api_key_value=config.api_key,
    )
    
    fs = project.get_feature_store()

    # Get the feature group
    fg = fs.get_feature_group(
        name=config.feature_group_name, 
        version=config.feature_group_version
    )
    
    # Retrieve the features
    fv = fs.get_or_create_feature_view(
        name = config.feature_view_name, 
        version = config.feature_view_version,
        query = fg.select_all(),
    )
    

    features: pd.DataFrame = fv.get_batch_data()
    return features
    # Convert the features to a DataFrame
    


if __name__ == '__main__':
    features = get_features_from_store()
    