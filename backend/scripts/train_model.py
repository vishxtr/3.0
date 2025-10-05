"""
Train a tiny RandomForest on synthetic URL-features for demonstration.
Saves model to ../../data/models/phish_model_v1.pkl and metadata.
"""
import random
import string
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SUSPICIOUS_WORDS = ["login", "secure", "update", "verify", "account", "bank", "confirm"]

def synth_url(has_suspicious=False):
    # generate synthetic urls: either benign or phishing-like
    domain = "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 12)))
    tld = random.choice([".com", ".net", ".io", ".org"])
    path_len = random.randint(0, 40)
    path = "".join(random.choices(string.ascii_lowercase + string.digits, k=path_len))
    if has_suspicious:
        # inject suspicious word randomly
        word = random.choice(SUSPICIOUS_WORDS)
        insert_at = random.randint(0, max(1, len(path)))
        path = path[:insert_at] + word + path[insert_at:]
    url = f"https://{domain}{tld}/{path}"
    # sometimes obfuscate with many digits
    if random.random() < 0.05:
        url += "/" + "".join(random.choices(string.digits, k=20))
    return url

def featurize(url):
    length = len(url)
    count_digits = sum(c.isdigit() for c in url)
    suspicious_count = sum(1 for w in SUSPICIOUS_WORDS if w in url.lower())
    dots = url.count(".")
    return [length, count_digits, suspicious_count, dots]

def generate_dataset(n=2000):
    rows = []
    for i in range(n):
        is_phish = random.random() < 0.35  # 35% phish in synthetic set
        url = synth_url(has_suspicious=is_phish)
        features = featurize(url)
        label = 1 if is_phish else 0
        rows.append({
            "url": url,
            "length": features[0],
            "count_digits": features[1],
            "suspicious_count": features[2],
            "dots": features[3],
            "label": label
        })
    return pd.DataFrame(rows)

def train_and_save():
    df = generate_dataset(2000)
    X = df[["length", "count_digits", "suspicious_count", "dots"]].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds).tolist()
    report = classification_report(y_test, preds, output_dict=True)
    model_path = MODELS_DIR / "phish_model_v1.pkl"
    joblib.dump(clf, model_path)
    metadata = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_samples": len(df),
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "classification_report": report,
        "features": ["length", "count_digits", "suspicious_count", "dots"]
    }
    meta_path = str(model_path).replace(".pkl", ".metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model -> {model_path} ({(model_path.stat().st_size/1024):.1f} KB)")
    print(f"Saved metadata -> {meta_path}")
    print(f"Training accuracy: {acc:.3f}")
    return model_path, meta_path, metadata

if __name__ == "__main__":
    train_and_save()