

build:
	docker build  -t kafka-to-store .

run-live: build
	docker run \
	-it --network redpanda_network \
	--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
	--env ONLINE_OFFLINE=online \
	--env KAFKA_INPUT_TOPIC=ohlc_live \
	--env AUTO_OFFSET_RESET=latest \
	--rm -p 8003:8003 \
	--name kafka-to-store-live kafka-to-store #First term is name of container, second is name of the image which is being pulled

run-backfill: build
	docker run \
	-it --network redpanda_network \
	--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
	--env ONLINE_OFFLINE=offline \
	--env KAFKA_INPUT_TOPIC=ohlc_backfill \
	--env AUTO_OFFSET_RESET=earliest \
	--rm -p 8003:8003 \
	--name kafka-to-store-historical kafka-to-store

lint:
	ruff check --fix

format:
	ruff format .