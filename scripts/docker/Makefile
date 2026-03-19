IMAGE           ?= python311-dev
DOCKERFILE      := Dockerfile.$(IMAGE)

GIT_BRANCH      := $(shell git branch --show-current)
SMLP_COMMIT     := $(shell git ls-remote https://github.com/SMLP-Systems/smlp.git refs/heads/$(GIT_BRANCH) | cut -f1)

.PHONY: build rebuild_smlp clean_build help

.DEFAULT_GOAL: build 

build: $(DOCKERFILE)
	env DOCKER_BUILDKIT=1 docker build \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
		--progress=plain \
		-f $(DOCKERFILE) \
		-t $(IMAGE) \
		.

rebuild_smlp: $(DOCKERFILE)
	env DOCKER_BUILDKIT=1 docker build \
		--build-arg CACHE_BUST_SMLP=$(SMLP_COMMIT) \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
		--progress=plain \
		-f $(DOCKERFILE) \
		-t $(IMAGE) \
		.

clean_build: $(DOCKERFILE)
	env DOCKER_BUILDKIT=1 docker build \
		--no-cache \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
		--progress=plain \
		-f $(DOCKERFILE) \
		-t $(IMAGE) \
		.

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
	@echo "  make build IMAGE=smlp-test-build-22.04-manylinux-python311"
	@echo ""
