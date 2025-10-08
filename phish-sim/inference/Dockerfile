# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Run the inference service
CMD ["python", "api/realtime_api.py", "--host", "0.0.0.0", "--port", "8004"]