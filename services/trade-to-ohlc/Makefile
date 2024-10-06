build:
	docker build -t ohlc .

run: build
	docker run \
	-it --network redpanda_network \
	--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
	--env OHLC_WINDOW_SECONDS=10 \
	--rm -p 8002:8002 \
	--name ohlc ohlc


lint:
	ruff check --fix

format:
	
	ruff format .