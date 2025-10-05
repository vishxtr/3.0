import joblib
import json
import os
from typing import Any, Dict
import numpy as np

class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.metadata = {}
        self.is_loaded = False

    def load(self):
        pkl = os.path.abspath(self.model_path)
        if not os.path.exists(pkl):
            raise FileNotFoundError(f"Model file not found: {pkl}")
        self.model = joblib.load(pkl)
        meta_path = pkl.replace(".pkl", ".metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
        self.is_loaded = True
        return self.model

    def predict_proba(self, X):
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        arr = np.array(X)
        return self.model.predict_proba(arr)