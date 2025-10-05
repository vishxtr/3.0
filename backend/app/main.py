from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .api import health, verdict
from .core.model_loader import ModelLoader
import os

app = FastAPI(title="PhishGuard Pro - Backend", version="0.1.0")

# CORS - allow all origins for hackathon/dev; recommend restricting in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach routers
app.include_router(health.router, prefix="/api")
app.include_router(verdict.router, prefix="/api")

# Load model singleton
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "models", "phish_model_v1.pkl")
MODEL_PATH = os.path.abspath(MODEL_PATH)
model_loader = ModelLoader(model_path=MODEL_PATH)

@app.on_event("startup")
def startup_event():
    try:
        model_loader.load()
        app.state.model = model_loader
    except Exception as e:
        # Keep server up; endpoints will return 503 if model not available
        print(f"[startup] model load warning: {e}")

class RootResponse(BaseModel):
    status: str

@app.get("/", response_model=RootResponse)
def root():
    return {"status": "PhishGuard Pro backend running"}