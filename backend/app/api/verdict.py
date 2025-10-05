from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
import time
import os

router = APIRouter()

class VerdictRequest(BaseModel):
    url: str
    source: Optional[str] = "web_ui"

def extract_features(url: str):
    # Lightweight feature extraction used at runtime (mirror features used in training)
    url = url.strip()
    length = len(url)
    count_digits = sum(c.isdigit() for c in url)
    suspicious_words = sum(word in url.lower() for word in ["login", "secure", "update", "verify", "account", "bank"])
    dots = url.count(".")
    return [length, count_digits, suspicious_words, dots]

@router.post("/verdict")
def get_verdict(req: VerdictRequest, request: Request):
    model_loader = request.app.state.model
    if not model_loader or not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    features = extract_features(req.url)
    proba = model_loader.predict_proba([features])[0]
    # Assuming proba[1] = probability of phishing label (0=legit,1=phish) from training
    phish_prob = float(proba[1])
    verdict = "phish" if phish_prob >= 0.5 else "legit"
    response = {
        "verdict": verdict,
        "confidence": round(phish_prob, 3),
        "features": {
            "length": features[0],
            "count_digits": features[1],
            "suspicious_words": features[2],
            "dots": features[3]
        },
        "meta": {"server_time": int(time.time())}
    }
    return response