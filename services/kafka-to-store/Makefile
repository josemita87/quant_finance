

build:
	docker build  -t kafka-to-store .

run-live: build
	docker run \
	-it --network redpanda_network \
	--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
	--env ONLINE_OFFLINE=online \
	--rm -p 8003:8003 \
	--name kafka-to-store-live kafka-to-store #First term is name of container, second is name of the image which is being pulled

run-backfill: build
	docker run \
	-it --network redpanda_network \
	--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
	--env ONLINE_OFFLINE=offline \
	--rm -p 8003:8003 \
	--name kafka-to-store-historical kafka-to-store

lint:
	ruff check --fix

format:
	ruff format .