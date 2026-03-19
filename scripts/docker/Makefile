SHELL           := /bin/bash
IMAGE           ?= smlp-test-build-22.04-manylinux-python311
DOCKERFILE      := Dockerfile.$(IMAGE)

GIT_BRANCH      := $(shell git branch --show-current)
SMLP_COMMIT     := $(shell git ls-remote https://github.com/SMLP-Systems/smlp.git refs/heads/$(GIT_BRANCH) | cut -f1)

WHEEL_SRC       := /app/smlp/dist/smlp-0.1.0-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
WHEEL_DST       := ../dist

define copy_out
	@if [[ "$(IMAGE)" == smlp-wheel* ]]; then \
		echo "Copying wheel from container image $(IMAGE)..."; \
		mkdir -p $(WHEEL_DST); \
		docker create --name smlp_tmp $(IMAGE); \
		docker cp smlp_tmp:$(WHEEL_SRC) $(WHEEL_DST); \
		docker rm smlp_tmp; \
	else \
		echo "Skipping copy_out: IMAGE=$(IMAGE) does not start with 'smlp-wheel'"; \
	fi
endef

.PHONY: build rebuild_smlp clean_build help
.DEFAULT_GOAL := build

build: $(DOCKERFILE)
	env DOCKER_BUILDKIT=1 docker build \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
		--progress=plain \
		-f $(DOCKERFILE) \
		-t $(IMAGE) \
		.
	$(copy_out)

rebuild_smlp: $(DOCKERFILE)
	env DOCKER_BUILDKIT=1 docker build \
		--build-arg CACHE_BUST_SMLP=$(SMLP_COMMIT) \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
		--progress=plain \
		-f $(DOCKERFILE) \
		-t $(IMAGE) \
		.
	$(copy_out)

clean_build: $(DOCKERFILE)
	env DOCKER_BUILDKIT=1 docker build \
		--no-cache \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
		--progress=plain \
		-f $(DOCKERFILE) \
		-t $(IMAGE) \
		.
	$(copy_out)

help:
	@echo ""
	@echo "Usage: make [target] [IMAGE=<image_name>]"
	@echo ""
	@echo "Targets:"
	@echo "  build          Normal build using Dockerfile.<IMAGE>  (default target)"
	@echo "  rebuild_smlp   Force re-fetch of smlp (cache-busted)"
	@echo "  clean_build    Full rebuild with --no-cache"
	@echo "  default image  $(IMAGE)"
	@echo "  help           Show this message"
	@echo ""
	@echo "Example:"
	@echo "  make build IMAGE=smlp-wheel_2_28-python311"
	@echo ""
