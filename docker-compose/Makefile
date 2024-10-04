
start-redpanda:
	docker-compose -f docker-compose.yml up -d

stop-redpanda:
	docker-compose -f docker-compose.yml down

build-feature-pipeline:
	docker compose -f feature-pipeline.yml build

#Feature pipeline with live data
run-feature-pipeline:
	docker compose -f feature-pipeline.yml up 

stop-feature-pipeline:
	docker compose -f feature-pipeline.yml down

#Feature pipeline with historical data
run-backfill-pipeline: build-feature-pipeline
	docker compose -f backfill-pipeline.yml up

stop-backfill-pipeline:
	docker compose -f backfill-pipeline.yml down