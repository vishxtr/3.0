# PhishGuard Pro (Hackathon Demo)
PhishGuard Pro — a lightweight, hackathon-ready phishing detection demo.  
**Built from scratch** for demo & judging. Uses synthetic data only.

## Quick start (development)
1. Create & activate venv:
   - Linux/macOS:
     ```
     python3 -m venv .venv
     . .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```
     python -m venv .venv
     .\\.venv\\Scripts\\Activate.ps1
     ```
2. Install & train:

```bash
pip install -r backend/requirements.txt
python backend/scripts/train_model.py
```

3. Run backend:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

4. Run frontend:

```bash
cd frontend
npm ci
npm run dev
```

## Docker

```bash
docker-compose up --build -d
```

backend -> http://localhost:8000
frontend -> http://localhost:5173

## Safety & ethics
- All data is synthetic. Do **not** use real victim or phishing payloads.
- This demo is for research/hackathon purposes only.