

build:
	docker build  -t trade-producer .

run: 
	docker run -it --network redpanda_network --rm -p 8001:8001 --name trade-producer trade-producer


lint:
	ruff check --fix

format:
	ruff format .