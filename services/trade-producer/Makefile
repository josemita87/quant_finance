

build:
	docker build --progress=plain -t trade-producer .

run: 
	docker run -it --network redpanda_network --rm -p 8001:8001 --name trade-producer trade-producer