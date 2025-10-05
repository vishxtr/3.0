#!/usr/bin/env python3
"""
Quick starter to prepare environment, train model, and start backend & frontend in dev mode.
"""
import os, subprocess, sys

def run(cmd, cwd=None):
    print(f"$ {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)

try:
    run("python3 -m venv .venv || python -m venv .venv")
    if os.name == "nt":
        activate = ".venv\\Scripts\\activate && "
    else:
        activate = ". .venv/bin/activate && "
    run(activate + "pip install --upgrade pip")
    run(activate + "pip install -r backend/requirements.txt")
    run(activate + "python backend/scripts/train_model.py")
    print("\\n✅ Model trained and saved to data/models/. You can now run backend with:")
    print("  . .venv/bin/activate && uvicorn backend.app.main:app --reload --port 8000")
    print("And run frontend dev server with:")
    print("  cd frontend && npm ci && npm run dev")
except Exception as e:
    print("Error during quick start:", e)
    sys.exit(1)