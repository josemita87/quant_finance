run:
	poetry run streamlit run src/frontend.py


build:
	docker build -t backend .
	
run-hops: build
	docker run -it \
		--env PROJECT_NAME=ml_system_c1 \
		--env FEATURE_GROUP_NAME=candles_fgap \
		--env FEATURE_GROUP_VERSION=1 \
		--env FEATURE_VIEW_VERSION=1 \
		--env FEATURE_VIEW_NAME=candles_fgap_view \
		--env-file .credentials.env \
		--name backend \
		backend


  project_name: str =  Field(..., env='PROJECT_NAME')
    feature_group_name: str = Field(..., env='FEATURE_GROUP_NAME')
    feature_group_version: int = Field(..., env='FEATURE_GROUP_VERSION')
    hopsworks_api_key:str = Field(..., env='HOPSWORKS_API_KEY')