# Phish-Sim Makefile
# Convenience commands for development and production

.PHONY: help deps test up down clean build e2e demo

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

test-e2e: ## Run end-to-end integration tests
	@echo "Running end-to-end integration tests..."
	python3 tests/test_e2e_integration.py

up: ## Start all services
	@echo "Starting services..."
	docker-compose up --build

up-backend: ## Start backend services only
	@echo "Starting backend services..."
	docker-compose up --build backend redis ml-service visual-service inference-service

up-frontend: ## Start frontend services only
	@echo "Starting frontend services..."
	docker-compose up --build frontend

up-monitoring: ## Start monitoring services
	@echo "Starting monitoring services..."
	docker-compose up --build prometheus elasticsearch kibana

down: ## Stop all services
	@echo "Stopping services..."
	docker-compose down

down-backend: ## Stop backend services only
	@echo "Stopping backend services..."
	docker-compose stop backend redis ml-service visual-service inference-service

down-frontend: ## Stop frontend services only
	@echo "Stopping frontend services..."
	docker-compose stop frontend

clean: ## Clean up containers and volumes
	@echo "Cleaning up..."
	docker-compose down -v --remove-orphans
	docker system prune -f

build: ## Build all services
	@echo "Building services..."
	docker-compose build

build-backend: ## Build backend services only
	@echo "Building backend services..."
	docker-compose build backend ml-service visual-service inference-service

build-frontend: ## Build frontend services only
	@echo "Building frontend services..."
	docker-compose build frontend

logs: ## Show logs for all services
	@echo "Showing logs..."
	docker-compose logs -f

logs-backend: ## Show backend logs
	@echo "Showing backend logs..."
	docker-compose logs -f backend redis ml-service visual-service inference-service

logs-frontend: ## Show frontend logs
	@echo "Showing frontend logs..."
	docker-compose logs -f frontend

dev: ## Start development environment
	@echo "Starting development environment..."
	docker-compose -f docker-compose.dev.yml up --build

demo: ## Run end-to-end integration demo
	@echo "Running end-to-end integration demo..."
	python3 demo_e2e_integration.py

demo-backend: ## Run backend demo
	@echo "Running backend demo..."
	python3 backend/demo_unified_api.py

demo-frontend: ## Run frontend demo
	@echo "Running frontend demo..."
	cd frontend && npm run dev

health: ## Check system health
	@echo "Checking system health..."
	curl -f http://localhost:8001/health || echo "Backend not available"
	curl -f http://localhost:3000/ || echo "Frontend not available"

status: ## Show system status
	@echo "System Status:"
	@echo "Backend API:"
	@curl -s http://localhost:8001/status | python3 -m json.tool || echo "Backend not available"
	@echo "Frontend:"
	@curl -s http://localhost:3000/ | head -5 || echo "Frontend not available"

restart: ## Restart all services
	@echo "Restarting services..."
	docker-compose restart

restart-backend: ## Restart backend services
	@echo "Restarting backend services..."
	docker-compose restart backend redis ml-service visual-service inference-service

restart-frontend: ## Restart frontend services
	@echo "Restarting frontend services..."
	docker-compose restart frontend

scale: ## Scale services (usage: make scale SERVICE=NUM)
	@echo "Scaling services..."
	docker-compose up --scale backend=2 --scale frontend=2

monitor: ## Open monitoring dashboards
	@echo "Opening monitoring dashboards..."
	@echo "Prometheus: http://localhost:9090"
	@echo "Kibana: http://localhost:5601"
	@echo "Elasticsearch: http://localhost:9200"