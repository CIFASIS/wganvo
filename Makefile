help:
	@echo "help  -- print this help"
	@echo "start -- start docker stack"
	@echo "stop  -- stop docker stack"
	@echo "ps    -- show status"
	@echo "top   -- displays the running processes"
	@echo "clean -- clean all artifacts"
	@echo "shell -- run bash inside docker"
	@echo "image  -- create ymy docker image"

start:
	docker-compose up -d

stop:
	docker-compose stop

ps:
	docker-compose ps
	
top:
	docker-compose top

clean: stop
	docker-compose rm --force -v

shell:
	docker exec -it wganvo-docker bash

image:
	docker-compose build

.PHONY: help start stop ps top clean shell image

