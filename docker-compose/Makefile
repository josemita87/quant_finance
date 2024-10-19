
start-redpanda:
	docker-compose -f redpanda.yml up -d
stop-redpanda:
	docker-compose -f redpanda.yml down

build-live:
	docker compose -f live-pipeline.yml build
build-backfill:
	docker compose -f backfill-pipeline.yml build

#Feature pipeline with live data
run-live: build-live	
	docker compose -f live-pipeline.yml up 
run-backfill: build-backfill
	docker compose -f backfill-pipeline.yml up

stop-live:
	docker compose -f live-pipeline.yml down
stop-backfill:
	docker compose -f backfill-pipeline.yml down	
	