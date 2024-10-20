

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
	--env PRODUCT_IDS='["DOT/USDT", "PEPE/EUR"]' \
	--env LAST_N_DAYS=15 \
	--env CACHE_DIR_PATH=/tmp/cached_backfill \
	--rm -p 8001:8001 \
	-v /Users/calcatscubedos.l./Desktop/ml_system_c1/services/trade-producer:/tmp/cached_backfill \
	--name trade-producer-backfill trade-producer



lint:
	ruff check --fix

format:
	ruff format .

dev:
	poetry run python src/main.py