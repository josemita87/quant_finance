

build:
	docker build  -t kafka-to-store .

run: build
	docker run \
	-it --network redpanda_network \
	--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
	--rm -p 8003:8003 \
	--name kafka-to-store kafka-to-store


lint:
	ruff check --fix

format:
	ruff format .