SHELL           := /bin/bash
IMAGE           ?= smlp-wheel_2_28-python311
DOCKERFILE      := Dockerfile.$(IMAGE)

GIT_BRANCH      := $(shell git branch --show-current)
SMLP_COMMIT     := $(shell git ls-remote https://github.com/SMLP-Systems/smlp.git refs/heads/$(GIT_BRANCH) | cut -f1)
RELEASE_TYPE    ?=
WHEEL_DST       := ../dist

ifeq ($(RELEASE_TYPE),test)
    RELEASE_BUILD_ARG := --build-arg RELEASE_TYPE=-test
else
    RELEASE_BUILD_ARG := --build-arg RELEASE_TYPE=
endif

define copy_out
	@if [[ "$(IMAGE)" == smlp-wheel* ]]; then \
		echo "Copying wheel from container image $(IMAGE)..."; \
		mkdir -p $(WHEEL_DST); \
		docker create --name smlp_tmp $(IMAGE); \
		docker cp smlp_tmp:/app/smlp/dist/. /tmp/smlp_dist_tmp; \
		docker rm smlp_tmp; \
                mv /tmp/smlp_dist_tmp/smlptech-*manylinux*.whl $(WHEEL_DST)/ ; \
                rm -rf /tmp/smlp_dist_tmp; \
	else \
		echo "Skipping copy_out: IMAGE=$(IMAGE) does not start with 'smlp-wheel'"; \
	fi
endef

.PHONY: build rebuild_smlp clean_build help
.DEFAULT_GOAL := build

build: $(DOCKERFILE)
	env DOCKER_BUILDKIT=1 docker build \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
		$(RELEASE_BUILD_ARG) \
		--progress=plain \
		-f $(DOCKERFILE) \
		-t $(IMAGE) \
		.
	$(copy_out)

rebuild_smlp: $(DOCKERFILE)
	env DOCKER_BUILDKIT=1 docker build \
		--build-arg CACHE_BUST_SMLP=$(SMLP_COMMIT) \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
		$(RELEASE_BUILD_ARG) \
		--progress=plain \
		-f $(DOCKERFILE) \
		-t $(IMAGE) \
		.
	$(copy_out)

clean_build: $(DOCKERFILE)
	env DOCKER_BUILDKIT=1 docker build \
		--no-cache \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
		$(RELEASE_BUILD_ARG) \
		--progress=plain \
		-f $(DOCKERFILE) \
		-t $(IMAGE) \
		.
	$(copy_out)

help:
	@echo    ""
	@echo    "Usage: make [target] [IMAGE=<image_name>] [SMLP_COMMIT=<tagname>"]
	@echo    ""
	@echo    "Targets:"
	@echo    "  build                               Normal build using Dockerfile.<IMAGE>  (default target)"
	@echo    "  rebuild_smlp                        Force smlp clone (docker cache busted)"
	@echo    "  rebuild_smlp SMLP_COMMIT=<tagname>  Force smlp clone (docker cache busted)"
	@echo -e "                                          \033[1mif the only change versus previus build is the new tag pushed to remote\033[0m"
	@echo    "  clean_build                         Full rebuild with --no-cache"
	@echo    "  default image $(IMAGE)"
	@echo    "  help                                Show this message"
	@echo    ""
	@echo    "Example:"
	@echo    "  make build IMAGE=python311-dev"
	@echo    ""
