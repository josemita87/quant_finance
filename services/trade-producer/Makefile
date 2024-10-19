

build:
	docker build  -t trade-producer .

run-live: build
	docker run \
	-it --network redpanda_network \
	--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
	--env KAFKA_OUTPUT_TOPIC=trade_live \
	--env LIVE_OR_HISTORICAL=live \
	--env PRODUCT_IDS='["BTC/EUR", "ETH/EUR"]' \
	--rm -p 8001:8001 \
	--name trade-producer-live trade-producer #First term is name of container, second is name of the image which is being pulled

run-backfill: build
	docker run \
	-it --network redpanda_network \
	--env KAFKA_BROKER_ADDRESS=redpanda-0:9092 \
	--env KAFKA_OUTPUT_TOPIC=trade_backfill \
	--env LIVE_OR_HISTORICAL=historical \
	--env PRODUCT_IDS='["BTC/EUR", "ETH/EUR"]' \
	--env LAST_N_DAYS=1 \
	--rm -p 8001:8001 \
	--name trade-producer-backfill trade-producer

lint:
	ruff check --fix

format:
	ruff format .

dev:
	poetry run python src/main.py