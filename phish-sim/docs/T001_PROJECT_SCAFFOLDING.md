# T001 - Project Scaffolding & CI

## Overview
This task establishes the foundational structure for the Phish-Sim project, including backend API, frontend application, Docker containerization, CI/CD pipeline, and comprehensive testing framework.

## Implementation Details

### Backend (FastAPI)
- **Framework**: FastAPI with Python 3.11
- **Features**: 
  - Health check endpoint
  - Phishing analysis API (placeholder)
  - Metrics endpoint
  - CORS middleware
  - Pydantic models for request/response validation
- **Testing**: pytest with comprehensive test coverage
- **Containerization**: Docker with health checks

### Frontend (React + TypeScript)
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS with dark theme
- **Features**:
  - Dashboard with system status
  - Analysis page for URL/text input
  - Simulator page (placeholder)
  - Settings page with system information
- **Testing**: Vitest with React Testing Library

### Infrastructure
- **Containerization**: Docker Compose with multiple environments
  - Development: Hot reload enabled
  - Testing: Isolated test environment
  - Production: Optimized builds
- **CI/CD**: GitHub Actions with:
  - Backend and frontend testing
  - Integration tests
  - Security scanning with Trivy
  - Automated deployment pipeline

### Project Structure
```
phish-sim/
├── backend/              # FastAPI backend service
│   ├── app/             # Application code
│   ├── tests/           # Backend tests
│   ├── requirements.txt # Python dependencies
│   └── Dockerfile       # Backend container
├── frontend/            # React frontend
│   ├── src/            # Source code
│   ├── public/         # Static assets
│   ├── package.json    # Node dependencies
│   └── Dockerfile      # Frontend container
├── docs/               # Documentation
├── tests/              # Integration tests
├── .github/workflows/  # CI/CD pipelines
├── docker-compose.yml  # Main compose file
├── Makefile           # Development commands
└── README.md          # Project documentation
```

## Testing Strategy

### Backend Tests
- Unit tests for all API endpoints
- Health check validation
- Request/response model validation
- Error handling tests

### Frontend Tests
- Component rendering tests
- Navigation tests
- User interaction tests
- Build validation

### Integration Tests
- End-to-end API testing
- Service communication validation
- Docker container health checks

## Security Considerations
- Non-root user in containers
- Security scanning in CI pipeline
- Input validation with Pydantic
- CORS configuration
- No sensitive data in codebase

## Performance Targets
- Backend response time: < 100ms
- Frontend build time: < 30s
- Container startup time: < 10s
- Test execution time: < 2 minutes

## Next Steps (T002)
- Implement data ingestion pipeline
- Create synthetic dataset generators
- Add public dataset integration
- Implement data preprocessing scripts

## Commands Reference

### Development
```bash
make deps      # Install dependencies
make up        # Start all services
make down      # Stop all services
make test      # Run all tests
make clean     # Clean up containers
```

### Docker Commands
```bash
docker compose up --build                    # Start with build
docker compose -f docker-compose.dev.yml up # Development mode
docker compose -f docker-compose.test.yml up # Test mode
```

### Testing
```bash
# Backend tests
cd backend && pytest

# Frontend tests
cd frontend && npm test

# Integration tests
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit
```

## API Documentation
Once running, visit:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health