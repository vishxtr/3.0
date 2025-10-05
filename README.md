# 🛡️ PhishGuard Pro

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.2.0-blue.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.118.0-green.svg)](https://fastapi.tiangolo.com/)

**PhishGuard Pro** — A modern, hackathon-ready phishing detection system with real-time URL scanning, ML-powered analysis, and an intuitive dashboard.

> ⚠️ **Demo Only**: This project uses synthetic data for demonstration purposes. Not intended for production use with real phishing data.

## ✨ Features

- 🔍 **Real-time URL Scanning**: Instant phishing detection with confidence scores
- 🤖 **ML-Powered Analysis**: RandomForest model trained on synthetic URL features
- 📊 **Interactive Dashboard**: Modern React UI with dark theme and animations
- 🐳 **Docker Ready**: Complete containerization for easy deployment
- 🧪 **Comprehensive Testing**: Backend API tests with pytest
- 📱 **Responsive Design**: Mobile-friendly interface with Tailwind CSS
- ⚡ **Fast Performance**: FastAPI backend with optimized ML inference

## 🚀 Quick Start

### Development Mode

1. **Setup Environment**:
   ```bash
   # Create virtual environment
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -r backend/requirements.txt
   cd frontend && npm install && cd ..
   ```

2. **Train ML Model**:
   ```bash
   python backend/scripts/train_model.py
   ```

3. **Start Services**:
   ```bash
   # Terminal 1: Backend
   uvicorn backend.app.main:app --reload --port 8000
   
   # Terminal 2: Frontend
   cd frontend && npm run dev
   ```

4. **Access Application**:
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Docker Deployment

```bash
# Build and start all services
docker-compose up --build -d

# Access the application
# Frontend: http://localhost:5173
# Backend: http://localhost:8000
```

## 🏗️ Architecture

```
phishguard-pro/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core business logic
│   │   └── main.py         # FastAPI application
│   ├── scripts/            # ML training scripts
│   └── tests/              # Backend tests
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # Reusable components
│   │   ├── pages/          # Page components
│   │   └── App.tsx         # Main app component
│   └── dist/               # Built frontend
├── data/                   # ML models and demo data
│   └── models/             # Trained ML models
└── docs/                   # Documentation
```

## 🧪 Testing

```bash
# Run backend tests
pytest backend/tests

# Run frontend tests
cd frontend && npm test
```

## 📊 API Endpoints

- `GET /api/health` - Health check
- `POST /api/verdict` - URL phishing analysis
  ```json
  {
    "url": "https://example.com/login",
    "source": "web_ui"
  }
  ```

## 🛡️ Safety & Ethics

- **Synthetic Data Only**: All training data is artificially generated
- **Demo Purpose**: Intended for hackathon demonstration only
- **No Real Phishing Data**: Never use actual phishing URLs or victim data
- **Research Use**: Suitable for educational and research purposes

## 🤝 Contributing

This is a hackathon project. For contributions or questions, please open an issue.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for hackathon demonstration**