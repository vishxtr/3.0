# Phish-Sim Makefile
# Convenience commands for development

.PHONY: help deps test up down clean build

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

deps: ## Install dependencies
	@echo "Installing dependencies..."
	docker-compose build

test: ## Run all tests
	@echo "Running tests..."
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit

test-backend: ## Run backend tests only
	@echo "Running backend tests..."
	docker-compose exec backend pytest

test-frontend: ## Run frontend tests only
	@echo "Running frontend tests..."
	docker-compose exec frontend npm test

up: ## Start all services
	@echo "Starting services..."
	docker-compose up --build

down: ## Stop all services
	@echo "Stopping services..."
	docker-compose down

clean: ## Clean up containers and volumes
	@echo "Cleaning up..."
	docker-compose down -v --remove-orphans
	docker system prune -f

build: ## Build all services
	@echo "Building services..."
	docker-compose build

logs: ## Show logs for all services
	@echo "Showing logs..."
	docker-compose logs -f

dev: ## Start development environment
	@echo "Starting development environment..."
	docker-compose -f docker-compose.dev.yml up --build