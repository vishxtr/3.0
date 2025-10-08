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

# Create data directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/synthetic

# Expose port (if needed for data service API)
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Run the data service
CMD ["python", "scripts/run_data_pipeline.py"]