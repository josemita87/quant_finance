build:
	docker build -t ohlc .

run-live: build
	docker run \
	-it --network redpanda_network \
	--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
	--env KAFKA_INPUT_TOPIC=trade_live \
	--rm -p 8002:8002 \
	--name ohlc ohlc

run-backfill: build
	docker run \
	-it --network redpanda_network \
	--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
	--env INPUT_TOPIC=trade_backfill \
	--rm -p 8002:8002 \
	--name ohlc ohlc


lint:
	ruff check --fix

format:
	
	ruff format .