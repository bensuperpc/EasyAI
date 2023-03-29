#//////////////////////////////////////////////////////////////
#//   ____                                                   //
#//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
#//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
#//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
#//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
#//                             |_|             |_|          //
#//////////////////////////////////////////////////////////////
#//                                                          //
#//  openNSFW, 2023                                           //
#//  Created: 19, April, 2022                                //
#//  Modified: 04, July, 2022                                //
#//  file: -                                                 //
#//  -                                                       //
#//  Source:                                                 //
#//  OS: ALL                                                 //
#//  CPU: ALL                                                //
#//                                                          //
#//////////////////////////////////////////////////////////////

PROJECT_NAME := openNSFW

# LANG := en
# LANG=$(LANG)

.PHONY: all
all: docker

.PHONY: build
build:
	docker buildx build --platform linux/amd64 -t bensuperpc/open_nsfw:latest .

.PHONY: start
build: build
	docker run -it --rm -v "$(shell pwd):/app" --workdir /app --user "$(shell id -u):$(shell id -g)" \
		bensuperpc/open_nsfw:latest

#   --security-opt no-new-privileges --cap-drop ALL --tmpfs /tmp:exec --tmpfs /run:exec \

.PHONY: docker
docker: start


.PHONY: cloc
cloc:
	cloc --fullpath --not-match-d="(build|.git|dataset)" --not-match-f="(.git)" .

.PHONY: update
update:
	git submodule update --recursive --remote
	git pull --recurse-submodules --all --progress

.PHONY: clean
clean:
	rm -rf build/*
