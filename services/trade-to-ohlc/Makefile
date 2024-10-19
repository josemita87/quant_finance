build:
	docker build -t ohlc .

run-live: build
	docker run \
	-it --network redpanda_network \
	--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
	--env KAFKA_INPUT_TOPIC=trade_live \
	--env KAFKA_OUTPUT_TOPIC=ohlc_live \
	--env AUTO_OFFSET_RESET=latest \
	--rm -p 8002:8002 \
	--name ohlc ohlc

run-backfill: build
	docker run \
	-it --network redpanda_network \
	--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
	--env KAFKA_INPUT_TOPIC=trade_backfill \
	--env KAFKA_OUTPUT_TOPIC=ohlc_backfill \
	--env AUTO_OFFSET_RESET=earliest \
	--rm -p 8002:8002 \
	--name ohlc ohlc


lint:
	ruff check --fix

format:
	
	ruff format .