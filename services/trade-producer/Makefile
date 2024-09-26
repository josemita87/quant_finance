

build:
	docker build  -t trade-producer .

run: build
	docker run \
	-it --network redpanda_network \
	--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
	--rm -p 8001:8001 \
	--name trade-producer trade-producer


lint:
	ruff check --fix

format:
	ruff format .