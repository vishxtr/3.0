# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Main FastAPI backend application for Phish-Sim
Real-Time AI/ML-Based Phishing Detection & Prevention — Web Simulation

This module imports and uses the unified API for all functionality.
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the unified API
from unified_api import app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)