# Production wrappers. The root Makefile is a crown jewel, so these targets
# live here. From the repo root:
#
#   make -f deploy/prod.mk prod-up
#   make -f deploy/prod.mk prod-down
#   make -f deploy/prod.mk prod-restart
#   make -f deploy/prod.mk prod-smoke
#
# Export QUANT_API_KEY and RESEARCH_API_KEY first. This file does not print
# them. SUDO prefixes docker and chown. Set SUDO= on the command line when
# docker does not need sudo; the compose recipes then call docker directly.

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
SUDO ?= sudo
COMPOSE_FILE := deploy/compose.prod.yml
PROJECT ?= dipcatcher-ops7
LAB_DATA ?= /tmp/dipcatcher-ops7-data/lab
RESEARCH_DATA ?= /tmp/dipcatcher-ops7-data/research
TLS_DIR ?= /tmp/dipcatcher-ops7-tls
LAB_TLS_PORT ?= 8443
RESEARCH_TLS_PORT ?= 8444
PROMETHEUS_PUBLISH_PORT ?= 9090
GRAFANA_PUBLISH_PORT ?= 3000
LAB_A ?= $(PROJECT)-lab-a-1
LAB_B ?= $(PROJECT)-lab-b-1
RESEARCH_A ?= $(PROJECT)-research-a-1
RESEARCH_B ?= $(PROJECT)-research-b-1
PRESERVE := QUANT_API_KEY,RESEARCH_API_KEY,DIPCATCHER_LAB_DATA,DIPCATCHER_RESEARCH_DATA,DIPCATCHER_TLS_DIR,LAB_TLS_PORT,RESEARCH_TLS_PORT,PROMETHEUS_PUBLISH_PORT,GRAFANA_PUBLISH_PORT

export DIPCATCHER_LAB_DATA := $(LAB_DATA)
export DIPCATCHER_RESEARCH_DATA := $(RESEARCH_DATA)
export DIPCATCHER_TLS_DIR := $(TLS_DIR)
export LAB_TLS_PORT
export RESEARCH_TLS_PORT
export PROMETHEUS_PUBLISH_PORT
export GRAFANA_PUBLISH_PORT
export SUDO
export PROJECT
export LAB_A
export LAB_B
export RESEARCH_A
export RESEARCH_B

ifeq ($(SUDO),)
COMPOSE := docker compose -f $(COMPOSE_FILE)
else
COMPOSE := $(SUDO) --preserve-env=$(PRESERVE) docker compose -f $(COMPOSE_FILE)
endif

.PHONY: prod-up prod-down prod-restart prod-smoke prod-certs prod-data prod-build

prod-up: prod-certs prod-data prod-build
	@test -n "$$QUANT_API_KEY" || { echo "QUANT_API_KEY is required" >&2; exit 1; }
	@test -n "$$RESEARCH_API_KEY" || { echo "RESEARCH_API_KEY is required" >&2; exit 1; }
	cd $(ROOT) && $(COMPOSE) up -d --no-build --remove-orphans

prod-down:
	@test -n "$$QUANT_API_KEY" || { echo "QUANT_API_KEY is required" >&2; exit 1; }
	@test -n "$$RESEARCH_API_KEY" || { echo "RESEARCH_API_KEY is required" >&2; exit 1; }
	cd $(ROOT) && $(COMPOSE) down --remove-orphans

prod-restart:
	@test -n "$$QUANT_API_KEY" || { echo "QUANT_API_KEY is required" >&2; exit 1; }
	@test -n "$$RESEARCH_API_KEY" || { echo "RESEARCH_API_KEY is required" >&2; exit 1; }
	cd $(ROOT) && uv run python scripts/rolling_restart.py \
		--containers $(LAB_A) $(LAB_B) \
		--probe-url http://127.0.0.1:8000/ready \
		--signal USR1 \
		--settle-seconds 5 \
		--key-env QUANT_API_KEY \
		--load-url https://127.0.0.1:$(LAB_TLS_PORT)/health \
		--ca-file $(TLS_DIR)/dev.crt \
		--load-seconds 110 \
		--load-concurrency 10 \
		--docker "$(SUDO) docker" \
		--zero-drops
	cd $(ROOT) && uv run python scripts/rolling_restart.py \
		--containers $(RESEARCH_A) $(RESEARCH_B) \
		--probe-url http://127.0.0.1:8010/ready \
		--signal USR1 \
		--settle-seconds 5 \
		--key-env RESEARCH_API_KEY \
		--load-url https://127.0.0.1:$(RESEARCH_TLS_PORT)/health \
		--ca-file $(TLS_DIR)/dev.crt \
		--load-seconds 110 \
		--load-concurrency 10 \
		--docker "$(SUDO) docker" \
		--zero-drops

prod-smoke:
	@cd $(ROOT) && uv run --quiet python scripts/prod_smoke.py

prod-certs:
	mkdir -p $(TLS_DIR)
	@if [ ! -s $(TLS_DIR)/dev.crt ] || [ ! -s $(TLS_DIR)/dev.key ]; then \
		openssl req -x509 -newkey rsa:2048 -sha256 -days 30 -nodes \
			-keyout $(TLS_DIR)/dev.key -out $(TLS_DIR)/dev.crt \
			-subj "/CN=localhost" \
			-addext "subjectAltName=DNS:localhost,IP:127.0.0.1"; \
	fi
	chmod 644 $(TLS_DIR)/dev.crt
	chmod 600 $(TLS_DIR)/dev.key

prod-data:
	$(SUDO) mkdir -p $(LAB_DATA) $(RESEARCH_DATA)
	$(SUDO) chown 10001:10001 $(LAB_DATA) $(RESEARCH_DATA)

prod-build:
	cd $(ROOT) && $(SUDO) docker build -t dipcatcher:local -f Dockerfile .
	cd $(ROOT) && $(SUDO) docker build -t dipcatcher-research-api -f docker/research-api.Dockerfile .
