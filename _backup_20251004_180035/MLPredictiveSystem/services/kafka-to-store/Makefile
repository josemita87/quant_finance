

build:
	docker build  -t kafka-to-store .

run-live: build
	docker run \
	-it --network redpanda_network \
	--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
	--env KAFKA_INPUT_TOPIC=ohlc_live \
	--env KAFKA_CONSUMER_GROUP=kafka-to-store-live \
	--env AUTO_OFFSET_RESET=latest \
	--env HOPSWORK_PROJECT_NAME=ml_system_c1 \
	--env HOPSWORK_FEATURE_GROUP_NAME=candles_fgap \
	--env HOPSWORK_FEATURE_GROUP_VERSION=1 \
	--env ONLINE_OFFLINE=online \
	--env BUFFER_SIZE=1 \
	--rm -p 8003:8003 \
	--env-file .credentials.env \
	--name kafka-to-store-live kafka-to-store #First term is name of container, second is name of the image which is being pulled

run-backfill: build
	docker run \
	-it --network redpanda_network \
	--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
	--env KAFKA_INPUT_TOPIC=ohlc_backfill \
	--env KAFKA_CONSUMER_GROUP=kafka-to-store-backfill \
	--env AUTO_OFFSET_RESET=earliest \
	--env HOPSWORK_PROJECT_NAME=ml_system_c1 \
	--env HOPSWORK_FEATURE_GROUP_NAME=candles_fgap \
	--env HOPSWORK_FEATURE_GROUP_VERSION=1 \
	--env ONLINE_OFFLINE=offline \
	--env BUFFER_SIZE=100 \
	--env TIMER=10 \
	--rm -p 8003:8003 \
	--env-file .credentials.env \
	--name kafka-to-store-historical kafka-to-store

lint:
	ruff check --fix

format:
	ruff format .