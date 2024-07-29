SHELL := /bin/bash

# import config. You can change the default config with `make cnf="config_special.env"
# build`
cnf ?= config.env
include $(cnf)
export $(shell sed 's/=.*//' $(cnf))


.PHONY: help
help: ## Show this help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

stop: ## Stop and remove a running container
	# docker stop $(APP_NAME); docker rm $(APP_NAME)
	@if [ "$(docker ps)" != "" ]; then \
		docker kill $(docker ps -aq); \
	fi
	@if [ "$(docker ps -aq)" != "" ]; then \
		docker rm $(docker ps -aq); \
	fi;

clear: stop ## Clear docker images and containers
	docker image prune -af;
	docker container prune -f;
	@if [ "$(docker images -q)" != "" ]; then \
		docker rmi $(docker images -q); \
	fi;

list: ## List docker images and containers
	docker image ls
	docker container ls -a \
		--filter status=exited \
		--filter status=paused \
		--filter status=dead \
		--filter status=created \
		--filter status=running

build: ## Build docker image
	./build.sh;

save_image_compressed: ## Save  the docker image as scalemae.tar file, and then compress to scalemae.tar.gz
	# If you can't install pigz on your system you can just have docker compress it, but
	# it's much slower:
	# docker save scalemae:latest | gzip > scalemae.tar.gz Use pigz to
	#
	# Parallelize the compression (apt install pigz):
	docker save scalemae:latest > scalemae.tar;
	tar --use-compress-program="pigz --best --recursive" \
		-cf scalemae.tar.gz scalemae.tar;
	# rm -f scalemae.tar;

run: ## Run the docker image
	sudo docker run -it \
		--rm \
		--network host \
		--gpus all \
		--name scalemae_container \
		-v /home/gbiamby/data/fmow:/proj/data \
		scalemae
#		/datasets/fmow_2024-01-10_1001/
#		# -v /home/gbiamby/proj/scale-mae:/proj/scalemae \

# test_knn_eval: ## Run the docker image
# 	docker run -it \
# 		--rm \
# 		--gpus all \
# 		--name scalemae_container \
# 		-mount source=./:target=/proj/scalemae/ \
# 		-v /datasets:/datasets \
# 		scalemae:latest bash /proj/scalemae/scripts/run_knn_eval.sh

size: ## Display docker image size
	docker history scalemae
