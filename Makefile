.PHONY: all setup train backend frontend build docker-up docker-down test

all: setup train build

setup:
	@echo "-> setting up backend venv & frontend deps"
	python3 -m venv .venv || python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r backend/requirements.txt
	cd frontend && npm ci

train:
	@echo "-> training model (synthetic)"
	. .venv/bin/activate && python backend/scripts/train_model.py

backend:
	. .venv/bin/activate && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

build:
	cd frontend && npm run build
	@echo "frontend build at frontend/dist"
	@echo "backend image built via docker (use make docker-up)"

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

test:
	. .venv/bin/activate && pytest -q backend/tests