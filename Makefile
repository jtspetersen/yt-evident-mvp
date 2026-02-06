# Evident Video Fact Checker - Makefile
# Docker Compose operations and shortcuts

.PHONY: help setup start stop restart status logs logs-app logs-ollama logs-searxng \
        run review shell models models-pull models-rm \
        clean clean-cache clean-all build build-no-cache web

# Default target
.DEFAULT_GOAL := help

# Docker compose directory
DOCKER_DIR := docker

# Detect GPU and set compose files
NVIDIA_GPU := $(shell nvidia-smi > /dev/null 2>&1 && echo 1 || echo 0)
AMD_GPU := $(shell rocm-smi > /dev/null 2>&1 && echo 1 || echo 0)

ifeq ($(NVIDIA_GPU),1)
    COMPOSE_FILES := -f $(DOCKER_DIR)/docker-compose.yml -f $(DOCKER_DIR)/docker-compose.gpu.yml
    GPU_TYPE := nvidia
else ifeq ($(AMD_GPU),1)
    COMPOSE_FILES := -f $(DOCKER_DIR)/docker-compose.yml -f $(DOCKER_DIR)/docker-compose.amd.yml
    GPU_TYPE := amd
else
    COMPOSE_FILES := -f $(DOCKER_DIR)/docker-compose.yml
    GPU_TYPE := none
endif

# ----------------------------
# Help
# ----------------------------

help: ## Show this help message
	@echo "Evident Video Fact Checker - Docker Operations"
	@echo ""
	@echo "Setup:"
	@echo "  make setup              - Run interactive setup wizard (first-time setup)"
	@echo ""
	@echo "Service Management:"
	@echo "  make start              - Start all services (ollama, searxng, redis)"
	@echo "  make stop               - Stop all services"
	@echo "  make restart            - Restart all services"
	@echo "  make status             - Show service status and health"
	@echo ""
	@echo "Logs:"
	@echo "  make logs               - Tail all service logs"
	@echo "  make logs-app           - Tail app logs only"
	@echo "  make logs-ollama        - Tail Ollama logs only"
	@echo "  make logs-searxng       - Tail SearXNG logs only"
	@echo ""
	@echo "Pipeline Execution (Native - recommended for Windows with GPU):"
	@echo "  make runvid ARGS='...'        - Run pipeline natively (uses local Ollama)"
	@echo "  make runvid-review ARGS='...' - Run with interactive review mode"
	@echo ""
	@echo "Pipeline Execution (Docker):"
	@echo "  make run ARGS='...'           - Run pipeline in Docker container"
	@echo "  make review ARGS='...'        - Run in Docker with review mode"
	@echo ""
	@echo "Web UI:"
	@echo "  make web                      - Start browser UI at http://localhost:8000"
	@echo ""
	@echo "Examples:"
	@echo "  make runvid ARGS='--infile inbox/transcript.txt --channel YourChannel'"
	@echo "  make runvid-review ARGS='--infile inbox/transcript.txt'"
	@echo ""
	@echo "Shell Access:"
	@echo "  make shell              - Open bash shell in app container"
	@echo ""
	@echo "Model Management:"
	@echo "  make models             - List all downloaded Ollama models"
	@echo "  make models-pull MODEL=<name>  - Pull/download a specific model"
	@echo "  make models-rm MODEL=<name>    - Remove a specific model"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean              - Remove stopped containers and volumes"
	@echo "  make clean-cache        - Clear URL cache (data/cache/)"
	@echo "  make clean-all          - DANGER: Remove ALL data including models"
	@echo ""
	@echo "Build:"
	@echo "  make build              - Rebuild app Docker image"
	@echo "  make build-no-cache     - Rebuild app image without cache"
	@echo ""
	@echo "GPU Status: $(if $(filter nvidia,$(GPU_TYPE)),✓ NVIDIA GPU detected,$(if $(filter amd,$(GPU_TYPE)),✓ AMD GPU detected (ROCm),✗ No GPU detected (CPU-only)))"

# ----------------------------
# Setup
# ----------------------------

setup: ## Run interactive setup wizard
	@echo "Running setup wizard..."
	python setup.py

# ----------------------------
# Service Management
# ----------------------------

start: ## Start all services
	@echo "Starting services (with $(if $(filter 1,$(GPU_AVAILABLE)),GPU,CPU-only) mode)..."
	docker compose $(COMPOSE_FILES) up -d

stop: ## Stop all services
	@echo "Stopping services..."
	docker compose $(COMPOSE_FILES) down

restart: stop start ## Restart all services

status: ## Show service status
	@echo "Service status:"
	@docker compose $(COMPOSE_FILES) ps

# ----------------------------
# Logs
# ----------------------------

logs: ## Tail all logs
	docker compose $(COMPOSE_FILES) logs -f

logs-app: ## Tail app logs only
	docker compose $(COMPOSE_FILES) logs -f app

logs-ollama: ## Tail Ollama logs only
	docker compose $(COMPOSE_FILES) logs -f ollama

logs-searxng: ## Tail SearXNG logs only
	docker compose $(COMPOSE_FILES) logs -f searxng

# ----------------------------
# Pipeline Execution
# ----------------------------

run: ## Run pipeline in Docker (use ARGS='...')
	@if [ -z "$(ARGS)" ]; then \
		echo "Usage: make run ARGS='--infile inbox/transcript.txt --channel YourChannel'"; \
		exit 1; \
	fi
	docker compose $(COMPOSE_FILES) run --rm app python -m app.main $(ARGS)

review: ## Run pipeline in Docker with review mode (use ARGS='...')
	@if [ -z "$(ARGS)" ]; then \
		echo "Usage: make review ARGS='--infile inbox/transcript.txt'"; \
		echo "Note: --review flag is added automatically"; \
		exit 1; \
	fi
	docker compose $(COMPOSE_FILES) run --rm app python -m app.main --review $(ARGS)

# Native execution (for Windows with native Ollama GPU)
runvid: ## Run pipeline natively (use ARGS='...')
	@if [ -z "$(ARGS)" ]; then \
		echo "Usage: make runvid ARGS='--infile inbox/transcript.txt --channel YourChannel'"; \
		exit 1; \
	fi
	. .venv/Scripts/activate && python -m app.main $(ARGS)

runvid-review: ## Run pipeline natively in review mode (use ARGS='...')
	@if [ -z "$(ARGS)" ]; then \
		echo "Usage: make runvid-review ARGS='--infile inbox/transcript.txt'"; \
		exit 1; \
	fi
	. .venv/Scripts/activate && python -m app.main --review $(ARGS)

web: ## Start web UI at http://localhost:8000
	. .venv/Scripts/activate && python -m app.web.server

# ----------------------------
# Shell Access
# ----------------------------

shell: ## Open bash in app container
	docker compose $(COMPOSE_FILES) run --rm app bash

# ----------------------------
# Model Management
# ----------------------------

models: ## List Ollama models
	@echo "Installed Ollama models:"
	@docker compose $(COMPOSE_FILES) exec ollama ollama list

models-pull: ## Pull/download a model (use MODEL=<name>)
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make models-pull MODEL=qwen3:8b"; \
		exit 1; \
	fi
	@echo "Pulling model: $(MODEL)"
	docker compose $(COMPOSE_FILES) exec ollama ollama pull $(MODEL)

models-rm: ## Remove a model (use MODEL=<name>)
	@if [ -z "$(MODEL)" ]; then \
		echo "Usage: make models-rm MODEL=qwen3:8b"; \
		exit 1; \
	fi
	@echo "Removing model: $(MODEL)"
	docker compose $(COMPOSE_FILES) exec ollama ollama rm $(MODEL)

# ----------------------------
# Cleanup
# ----------------------------

clean: ## Remove stopped containers and unused volumes
	@echo "Cleaning up stopped containers and volumes..."
	docker compose $(COMPOSE_FILES) down -v
	@echo "Removing unused Docker resources..."
	docker system prune -f

clean-cache: ## Clear URL cache
	@echo "WARNING: This will delete all cached web pages."
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "Clearing URL cache..."; \
		rm -rf data/cache/*; \
		echo "Cache cleared."; \
	else \
		echo "Cancelled."; \
	fi

clean-all: ## DANGER: Remove ALL data including models
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║                            DANGER                                 ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "This will delete:"
	@echo "  - All Docker containers and volumes"
	@echo "  - All Ollama models (will need to re-download)"
	@echo "  - All cached web pages"
	@echo "  - All run outputs"
	@echo "  - All logs"
	@echo ""
	@echo "This will NOT delete:"
	@echo "  - inbox/ (your transcripts)"
	@echo "  - .env (your configuration)"
	@echo ""
	@read -p "Type 'DELETE' to confirm: " confirm; \
	if [ "$$confirm" = "DELETE" ]; then \
		echo "Removing all data..."; \
		docker compose $(COMPOSE_FILES) down -v; \
		rm -rf data/cache/* data/runs/* data/store/* data/logs/* data/ollama/*; \
		echo "All data removed. Run 'make setup' to reinitialize."; \
	else \
		echo "Cancelled."; \
	fi

# ----------------------------
# Build
# ----------------------------

build: ## Rebuild app Docker image
	@echo "Building app image..."
	docker compose $(COMPOSE_FILES) build app

build-no-cache: ## Rebuild app image without cache
	@echo "Building app image (no cache)..."
	docker compose $(COMPOSE_FILES) build --no-cache app
