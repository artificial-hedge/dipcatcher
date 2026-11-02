# Supply-chain targets for the two runtime images.
#
# The root Makefile is a crown jewel, so this file is the target that runs
# today:
#
#   make -f deploy/supply.mk sbom
#
# `make sbom` from the repo root is the owner step in pr_body.md (a one-target
# delegate into this file). Run from any directory; paths are anchored here.
#
# Images:
#   dipcatcher:supply                  root Dockerfile
#   dipcatcher:supply-hardened         deploy/lab.Dockerfile (numeric user,
#                                      JSON healthcheck; same locked venv)
#   dipcatcher-research-api:supply     docker/research-api.Dockerfile

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
DOCKER ?= docker
SYFT ?= syft
IMAGE_LAB ?= dipcatcher:supply
IMAGE_LAB_HARDENED ?= dipcatcher:supply-hardened
IMAGE_RESEARCH ?= dipcatcher-research-api:supply
SBOM_DIR ?= $(ROOT)/.tmp_sbom

.PHONY: sbom build-images sbom-files check-image-lock

sbom: build-images sbom-files check-image-lock ## CycloneDX SBOMs and uv.lock parity

build-images: ## Build both Dockerfiles and the lab harden layer
	$(DOCKER) build -t $(IMAGE_LAB) -f $(ROOT)/Dockerfile $(ROOT)
	$(DOCKER) build -t $(IMAGE_LAB_HARDENED) \
		--build-arg BASE=$(IMAGE_LAB) \
		-f $(ROOT)/deploy/lab.Dockerfile $(ROOT)
	$(DOCKER) build -t $(IMAGE_RESEARCH) \
		-f $(ROOT)/docker/research-api.Dockerfile $(ROOT)

sbom-files: ## syft CycloneDX JSON for each runtime image
	mkdir -p $(SBOM_DIR)
	$(SYFT) $(IMAGE_LAB) -o cyclonedx-json=$(SBOM_DIR)/dipcatcher.cdx.json
	$(SYFT) $(IMAGE_LAB_HARDENED) -o cyclonedx-json=$(SBOM_DIR)/dipcatcher-hardened.cdx.json
	$(SYFT) $(IMAGE_RESEARCH) -o cyclonedx-json=$(SBOM_DIR)/dipcatcher-research-api.cdx.json

check-image-lock: ## Fail if an image's Python dists drift from uv.lock
	uv run python $(ROOT)/scripts/check_image_python_lock.py \
		--root $(ROOT) \
		--lock $(ROOT)/uv.lock \
		--image $(IMAGE_LAB) \
		--image $(IMAGE_RESEARCH)
