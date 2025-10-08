# Real-Time AI/ML-Based Phishing Detection & Prevention — Web Simulation

A fully-featured web application that simulates realistic phishing attack flows, detection pipelines, visualization and management consoles, and adversarial testing.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Make (optional, for convenience commands)

### Running the Application

1. **Clone and navigate to the project:**
   ```bash
   cd /workspace/phish-sim
   ```

2. **Start all services:**
   ```bash
   docker-compose up --build
   ```
   
   Or using Make:
   ```bash
   make up
   ```

3. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Development Commands

```bash
# Install dependencies
make deps

# Run tests
make test

# Start services
make up

# Stop services
make down
```

## Project Structure

```
phish-sim/
├── backend/          # FastAPI backend service
├── frontend/         # React + TypeScript frontend
├── ml/              # ML models and pipelines
├── simulator/       # Phishing simulation components
├── data/            # Datasets and synthetic data generators
├── tests/           # Integration and E2E tests
├── docs/            # Documentation
├── ci/              # CI/CD configuration
└── .github/workflows/ # GitHub Actions
```

## Features

- **Real-time Phishing Detection**: Multi-modal analysis (URL, text, visual)
- **Interactive Dashboard**: Dark-themed UI with live threat monitoring
- **Sandbox Simulation**: Headless browser testing with screenshot capture
- **Adversarial Testing**: Red-team simulation and zero-day detection
- **Explainable AI**: Detailed reasoning for each detection decision
- **Performance Metrics**: Sub-50ms inference with comprehensive benchmarking

## Technology Stack

- **Backend**: Python 3.11, FastAPI, SQLite
- **Frontend**: React, TypeScript, Vite, Tailwind CSS
- **ML**: PyTorch, Transformers, NetworkX
- **Testing**: Playwright, pytest, vitest
- **Containerization**: Docker, docker-compose

## License

MIT License - see LICENSE file for details.

## Contributing

This is a simulation project for educational and research purposes. All data is synthetic or from public datasets only.